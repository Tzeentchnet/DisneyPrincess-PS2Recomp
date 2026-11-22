#include "iop_spu2.h"

#include <algorithm>
#include <cstring>

namespace ps2x::iop::detail
{
    namespace
    {
        // Register offsets within a core.
        constexpr uint32_t kCoreSize = 0x400u;
        constexpr uint32_t kVoiceStride = 0x10u;
        constexpr uint32_t kPitch = 0x004u;
        constexpr uint32_t kEnvx = 0x00Au;
        constexpr uint32_t kKeyOn = 0x1A0u;
        constexpr uint32_t kKeyOff = 0x1A4u;
        constexpr uint32_t kTransferAddress = 0x1A8u;
        constexpr uint32_t kTransferData = 0x1ACu;
        constexpr uint32_t kAutoDma = 0x1B0u;
        constexpr uint32_t kAddresses = 0x1C0u;
        constexpr uint32_t kAddressStride = 12u;
        constexpr uint32_t kAddressesEnd = 0x2E0u;
        constexpr uint32_t kStart = 0u;
        constexpr uint32_t kLoop = 4u;
        constexpr uint32_t kNext = 8u;
        constexpr uint32_t kEndx = 0x340u;
        constexpr uint32_t kStatx = 0x344u;

        constexpr uint16_t kStatxDmaReady = 0x80u;
        constexpr uint16_t kEnvelopeMax = 0x7FFFu;
        constexpr uint32_t kMaxPitch = 0x3FFFu;
        constexpr uint32_t kBlockWords = 8u;
        constexpr uint32_t kBlockSamples = 28u;
        constexpr uint32_t kBlockPhase = kBlockSamples << 12u;
        constexpr uint32_t kWordMask = IopSpu2::RamBytes / 2u - 1u;

        constexpr uint8_t kFlagEnd = 1u;
        constexpr uint8_t kFlagRepeat = 2u;
        constexpr uint8_t kFlagLoopStart = 4u;
    }

    IopSpu2::IopSpu2()
        : m_ram(RamBytes)
    {
    }

    void IopSpu2::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint8_t{0});
        m_regs.fill(0u);
        m_voices = {};
    }

    uint64_t IopSpu2::sampleNow() const noexcept
    {
        return m_clock ? *m_clock / IopCyclesPerSample : 0u;
    }

    uint16_t &IopSpu2::reg(uint32_t core, uint32_t offset)
    {
        return m_regs[(core * kCoreSize + offset) / 2u];
    }

    uint32_t IopSpu2::address(uint32_t core, uint32_t offset)
    {
        return ((static_cast<uint32_t>(reg(core, offset)) << 16u) | reg(core, offset + 2u)) & kWordMask;
    }

    void IopSpu2::setAddress(uint32_t core, uint32_t offset, uint32_t value)
    {
        reg(core, offset) = static_cast<uint16_t>((value & kWordMask) >> 16u);
        reg(core, offset + 2u) = static_cast<uint16_t>(value);
    }

    void IopSpu2::enterBlock(uint32_t core, uint32_t index)
    {
        const Voice &voice = m_voices[core][index];
        const uint8_t flags = m_ram[((voice.block & kWordMask) * 2u + 1u) & (RamBytes - 1u)];
        if ((flags & kFlagLoopStart) != 0u && !voice.explicitLoop)
            setAddress(core, kAddresses + index * kAddressStride + kLoop, voice.block);
    }

    void IopSpu2::advance(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        const uint64_t now = sampleNow();
        const uint64_t elapsed = now > voice.sample ? now - voice.sample : 0u;
        voice.sample = now;
        if (!voice.playing || elapsed == 0u)
            return;

        const uint32_t addresses = kAddresses + index * kAddressStride;
        const uint32_t registers = index * kVoiceStride;
        const uint64_t pitch = std::min<uint32_t>(reg(core, registers + kPitch), kMaxPitch);

        uint64_t phase = voice.phase + elapsed * pitch;
        while (phase >= kBlockPhase)
        {
            phase -= kBlockPhase;
            const uint8_t flags = m_ram[(voice.block * 2u + 1u) & (RamBytes - 1u)];
            if ((flags & kFlagEnd) != 0u)
            {
                reg(core, kEndx + (index >= 16u ? 2u : 0u)) |= static_cast<uint16_t>(1u << (index & 15u));
                voice.block = address(core, addresses + kLoop);
                if ((flags & kFlagRepeat) == 0u)
                {
                    voice.playing = false;
                    reg(core, registers + kEnvx) = 0u;
                    phase = 0u;
                    break;
                }
            }
            else
            {
                voice.block = (voice.block + kBlockWords) & kWordMask;
            }
            enterBlock(core, index);
        }
        voice.phase = static_cast<uint32_t>(phase);

        // One header word, then four samples to a word.
        setAddress(core, addresses + kNext, voice.block + 1u + ((voice.phase >> 12u) / 4u));
    }

    void IopSpu2::advanceCore(uint32_t core)
    {
        for (uint32_t index = 0u; index < kVoices; ++index)
            advance(core, index);
    }

    void IopSpu2::keyOn(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        const uint32_t addresses = kAddresses + index * kAddressStride;
        voice = {};
        voice.block = address(core, addresses + kStart) & ~(kBlockWords - 1u);
        voice.sample = sampleNow();
        voice.playing = true;

        reg(core, index * kVoiceStride + kEnvx) = kEnvelopeMax;
        reg(core, kEndx + (index >= 16u ? 2u : 0u)) &= static_cast<uint16_t>(~(1u << (index & 15u)));
        setAddress(core, addresses + kNext, voice.block + 1u);
        enterBlock(core, index);
    }

    void IopSpu2::keyOff(uint32_t core, uint32_t index)
    {
        advance(core, index);
        // ADSR is not modelled, so a release is immediate.
        m_voices[core][index].playing = false;
        reg(core, index * kVoiceStride + kEnvx) = 0u;
    }

    uint16_t IopSpu2::read16(uint32_t physical)
    {
        const uint32_t relative = (physical - Base) & ~1u;
        const uint32_t core = relative / kCoreSize;
        const uint32_t offset = relative % kCoreSize;

        if (offset < kVoices * kVoiceStride && offset % kVoiceStride == kEnvx)
            advance(core, offset / kVoiceStride);
        else if (offset >= kAddresses && offset < kAddressesEnd)
            advance(core, (offset - kAddresses) / kAddressStride);
        else if (offset == kEndx || offset == kEndx + 2u)
            advanceCore(core);
        return reg(core, offset);
    }

    void IopSpu2::write16(uint32_t physical, uint16_t value)
    {
        const uint32_t relative = (physical - Base) & ~1u;
        const uint32_t core = relative / kCoreSize;
        const uint32_t offset = relative % kCoreSize;

        if (offset < kVoices * kVoiceStride && offset % kVoiceStride == kPitch)
        {
            // The old pitch applies up to now.
            advance(core, offset / kVoiceStride);
        }
        else if (offset >= kAddresses && offset < kAddressesEnd)
        {
            const uint32_t index = (offset - kAddresses) / kAddressStride;
            const uint32_t field = (offset - kAddresses) % kAddressStride;
            advance(core, index);
            if (field >= kNext)
                return; // NAX is the hardware's.
            if (field >= kLoop)
                m_voices[core][index].explicitLoop = true;
        }
        else if (offset == kKeyOn || offset == kKeyOn + 2u || offset == kKeyOff || offset == kKeyOff + 2u)
        {
            reg(core, offset) = value;
            const uint32_t first = (offset & 2u) != 0u ? 16u : 0u;
            for (uint32_t bit = 0u; bit < 16u && first + bit < kVoices; ++bit)
            {
                if ((value & (1u << bit)) == 0u)
                    continue;
                if (offset < kKeyOff)
                    keyOn(core, first + bit);
                else
                    keyOff(core, first + bit);
            }
            return;
        }
        else if (offset == kEndx || offset == kEndx + 2u)
        {
            // Any write clears that half of ENDX.
            advanceCore(core);
            reg(core, offset) = 0u;
            return;
        }
        else if (offset == kTransferData)
        {
            const uint32_t target = address(core, kTransferAddress);
            for (uint32_t other = 0u; other < kCores; ++other)
                advanceCore(other);
            std::memcpy(m_ram.data() + target * 2u, &value, sizeof(value));
            setAddress(core, kTransferAddress, target + 1u);
            return;
        }
        reg(core, offset) = value;
    }

    void IopSpu2::dmaWrite(uint32_t core, std::span<const uint8_t> bytes)
    {
        // With AutoDMA the data is a PCM input block and never lands in sample RAM.
        if ((reg(core, kAutoDma) & (1u << core)) != 0u)
            return;

        // Voices may already have played past blocks that are being replaced.
        for (uint32_t other = 0u; other < kCores; ++other)
            advanceCore(other);

        uint32_t target = address(core, kTransferAddress) * 2u;
        for (size_t done = 0u; done < bytes.size();)
        {
            const size_t count = std::min<size_t>(bytes.size() - done, RamBytes - target);
            std::memcpy(m_ram.data() + target, bytes.data() + done, count);
            done += count;
            target = static_cast<uint32_t>(target + count) & (RamBytes - 1u);
        }
    }

    void IopSpu2::completeDma(uint32_t core)
    {
        reg(core, kStatx) |= kStatxDmaReady;
    }
}
