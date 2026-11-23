#include "iop_spu2.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace ps2x::iop::detail
{
    namespace
    {
        // ADPCM and envelope semantics were informed by ps2-web-recomp.
        // See THIRD_PARTY_NOTICES.md for its MIT license.
        constexpr uint32_t kCoreSize = 0x400u;
        constexpr uint32_t kVoiceStride = 0x10u;
        constexpr uint32_t kVolumeLeft = 0x000u;
        constexpr uint32_t kVolumeRight = 0x002u;
        constexpr uint32_t kPitch = 0x004u;
        constexpr uint32_t kAdsr1 = 0x006u;
        constexpr uint32_t kAdsr2 = 0x008u;
        constexpr uint32_t kEnvx = 0x00Au;
        constexpr uint32_t kCurrentVolumeLeft = 0x00Cu;
        constexpr uint32_t kCurrentVolumeRight = 0x00Eu;
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

        constexpr uint32_t kMasterBase = 0x760u;
        constexpr uint32_t kMasterStride = 0x28u;
        constexpr uint32_t kMasterEnd = kMasterBase + 2u * kMasterStride;
        constexpr uint32_t kMasterVolumeLeft = 0x00u;
        constexpr uint32_t kMasterVolumeRight = 0x02u;
        constexpr uint32_t kInputVolumeLeft = 0x0Cu;
        constexpr uint32_t kInputVolumeRight = 0x0Eu;

        constexpr uint16_t kStatxDmaReady = 0x80u;
        constexpr uint32_t kMaxPitch = 0x3FFFu;
        constexpr uint32_t kBlockWords = 8u;
        constexpr uint32_t kBlockSamples = 28u;
        constexpr uint32_t kBlockPhase = kBlockSamples << 12u;
        constexpr uint32_t kWordMask = IopSpu2::RamBytes / 2u - 1u;
        constexpr uint32_t kByteMask = IopSpu2::RamBytes - 1u;
        constexpr size_t kAutoDmaBlockBytes = 1024u;

        constexpr uint8_t kFlagEnd = 1u;
        constexpr uint8_t kFlagRepeat = 2u;
        constexpr uint8_t kFlagLoopStart = 4u;

        constexpr int32_t kAdpcmCoefficients[5][2] = {
            {0, 0},
            {60, 0},
            {115, -52},
            {98, -55},
            {122, -60},
        };

        int16_t clampSample(int64_t value)
        {
            return static_cast<int16_t>(
                std::clamp<int64_t>(value,
                                    std::numeric_limits<int16_t>::min(),
                                    std::numeric_limits<int16_t>::max()));
        }

        int32_t fixedVolume(uint16_t value)
        {
            return static_cast<int16_t>(static_cast<uint16_t>(value << 1u));
        }

        int32_t masterVolume(uint16_t value)
        {
            return (value & 0x8000u) != 0u ? 0x7FFF : fixedVolume(value);
        }
    }

    IopSpu2::IopSpu2()
        : m_ram(RamBytes)
    {
        const char *const trace = std::getenv("PS2X_TRACE_AUDIO");
        m_trace = trace != nullptr && trace[0] != '\0' && trace[0] != '0';
        m_audio.reserve(4096u);
        reset();
    }

    void IopSpu2::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint8_t{0});
        m_regs.fill(0u);
        m_voices = {};
        for (auto &queue : m_autoDma)
            queue.clear();
        for (uint32_t core = 0u; core < kCores; ++core)
        {
            m_masterVolume[core] = {0x3FFFu, 0x3FFFu};
            m_inputVolume[core] = {0x7FFFu, 0x7FFFu};
        }
        m_audio.clear();
        m_renderedSample = 0u;
        m_audioStarted = false;
        m_traceFrames = 0u;
        m_autoDmaUnderflowFrames = {};
        m_autoDmaUnderflowRuns = {};
        m_autoDmaMinimumDepth.fill(std::numeric_limits<size_t>::max());
        m_autoDmaMaximumDepth = {};
        m_autoDmaUnderflowActive = {};
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

    void IopSpu2::decodeBlock(uint32_t core, uint32_t index, uint32_t slot)
    {
        Voice &voice = m_voices[core][index];
        const uint32_t block = voice.decodeAddress & kWordMask;
        const uint32_t byteAddress = block * 2u;
        const uint8_t header = m_ram[byteAddress & kByteMask];
        const uint8_t flags = m_ram[(byteAddress + 1u) & kByteMask];
        const uint32_t shift = std::min<uint32_t>(header & 0x0Fu, 12u);
        const uint32_t filter = std::min<uint32_t>((header >> 4u) & 0x0Fu, 4u);

        int16_t *destination = voice.decoded.data() + slot * kBlockSamples;
        for (uint32_t sample = 0u; sample < kBlockSamples; ++sample)
        {
            const uint8_t packed = m_ram[(byteAddress + 2u + sample / 2u) & kByteMask];
            int32_t nibble = (sample & 1u) != 0u ? packed >> 4u : packed & 0x0Fu;
            if ((nibble & 8) != 0)
                nibble -= 16;

            int32_t decoded = nibble * (1 << (12u - shift));
            decoded += (voice.history1 * kAdpcmCoefficients[filter][0] +
                        voice.history2 * kAdpcmCoefficients[filter][1] + 32) >>
                       6;
            decoded = std::clamp(decoded, -32768, 32767);
            voice.history2 = voice.history1;
            voice.history1 = decoded;
            destination[sample] = static_cast<int16_t>(decoded);
        }

        voice.blocks[slot] = block;
        voice.flags[slot] = flags;
        if ((flags & kFlagLoopStart) != 0u && !voice.explicitLoop)
            setAddress(core, kAddresses + index * kAddressStride + kLoop, block);
        voice.decodeAddress = (flags & kFlagEnd) != 0u
                                  ? address(core, kAddresses + index * kAddressStride + kLoop)
                                  : (block + kBlockWords) & kWordMask;
    }

    void IopSpu2::stepEnvelope(Envelope &envelope,
                               uint32_t rate,
                               bool decrease,
                               bool exponential)
    {
        const uint32_t shift = rate >> 2u;
        uint32_t interval = 1u << (shift > 11u ? shift - 11u : 0u);
        if (exponential && !decrease && envelope.level > 0x6000)
            interval *= 4u;
        if (++envelope.counter < interval)
            return;

        envelope.counter = 0u;
        const int32_t magnitude = decrease ? -8 + static_cast<int32_t>(rate & 3u)
                                           : 7 - static_cast<int32_t>(rate & 3u);
        int32_t delta = magnitude * (1 << (shift < 11u ? 11u - shift : 0u));
        if (exponential && decrease)
            delta = static_cast<int32_t>(
                (static_cast<int64_t>(delta) * envelope.level) >> 15u);
        envelope.level = std::clamp(envelope.level + delta, 0, 0x7FFF);
    }

    void IopSpu2::updateEnvelope(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        const uint32_t registers = index * kVoiceStride;
        const uint16_t adsr1 = reg(core, registers + kAdsr1);
        const uint16_t adsr2 = reg(core, registers + kAdsr2);

        switch (voice.envelopePhase)
        {
        case EnvelopePhase::Attack:
            stepEnvelope(voice.envelope,
                         (adsr1 >> 8u) & 0x7Fu,
                         false,
                         (adsr1 & 0x8000u) != 0u);
            if (voice.envelope.level >= 0x7FFF)
                voice.envelopePhase = EnvelopePhase::Decay;
            break;
        case EnvelopePhase::Decay:
            stepEnvelope(voice.envelope, ((adsr1 >> 4u) & 0x0Fu) << 2u, true, true);
            if (voice.envelope.level <=
                std::min<int32_t>(((adsr1 & 0x0Fu) + 1u) * 0x800u, 0x7FFF))
            {
                voice.envelopePhase = EnvelopePhase::Sustain;
            }
            break;
        case EnvelopePhase::Sustain:
            stepEnvelope(voice.envelope,
                         (adsr2 >> 6u) & 0x7Fu,
                         (adsr2 & 0x4000u) != 0u,
                         (adsr2 & 0x8000u) != 0u);
            break;
        case EnvelopePhase::Release:
            stepEnvelope(voice.envelope,
                         (adsr2 & 0x1Fu) << 2u,
                         true,
                         (adsr2 & 0x20u) != 0u);
            if (voice.envelope.level == 0)
            {
                voice.envelopePhase = EnvelopePhase::Off;
                voice.playing = false;
            }
            break;
        case EnvelopePhase::Off:
            break;
        }
        reg(core, registers + kEnvx) = static_cast<uint16_t>(voice.envelope.level);
    }

    int32_t IopSpu2::voiceVolume(uint16_t value, Envelope &envelope)
    {
        if ((value & 0x8000u) == 0u)
            return fixedVolume(value);
        stepEnvelope(envelope,
                     value & 0x7Fu,
                     (value & 0x2000u) != 0u,
                     (value & 0x4000u) != 0u);
        return envelope.level;
    }

    void IopSpu2::updateNextAddress(uint32_t core, uint32_t index)
    {
        const Voice &voice = m_voices[core][index];
        const uint32_t next = voice.blocks[0] + 1u + ((voice.phase >> 12u) / 4u);
        setAddress(core, kAddresses + index * kAddressStride + kNext, next);
    }

    int32_t IopSpu2::renderVoice(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        if (!voice.playing)
            return 0;

        const uint32_t registers = index * kVoiceStride;
        const uint32_t sampleIndex = voice.phase >> 12u;
        const uint32_t fraction = voice.phase & 0x0FFFu;
        const int32_t decoded =
            (voice.decoded[sampleIndex] * static_cast<int32_t>(0x1000u - fraction) +
             voice.decoded[sampleIndex + 1u] * static_cast<int32_t>(fraction)) >>
            12u;

        voice.phase += std::min<uint32_t>(reg(core, registers + kPitch), kMaxPitch);
        while (voice.phase >= kBlockPhase)
        {
            voice.phase -= kBlockPhase;
            const uint8_t completedFlags = voice.flags[0];
            if ((completedFlags & kFlagEnd) != 0u)
            {
                reg(core, kEndx + (index >= 16u ? 2u : 0u)) |=
                    static_cast<uint16_t>(1u << (index & 15u));
            }

            std::move(voice.decoded.begin() + kBlockSamples,
                      voice.decoded.end(),
                      voice.decoded.begin());
            voice.blocks[0] = voice.blocks[1];
            voice.flags[0] = voice.flags[1];

            if ((completedFlags & kFlagEnd) != 0u &&
                (completedFlags & kFlagRepeat) == 0u)
            {
                voice.playing = false;
                voice.envelopePhase = EnvelopePhase::Off;
                voice.envelope.level = 0;
                voice.phase = 0u;
                break;
            }
            decodeBlock(core, index, 1u);
        }

        updateEnvelope(core, index);
        updateNextAddress(core, index);
        return static_cast<int32_t>(
            (static_cast<int64_t>(decoded) * voice.envelope.level) >> 15u);
    }

    IopSpu2::StereoFrame IopSpu2::renderAutoDma(uint32_t core)
    {
        const bool enabled = (reg(core, kAutoDma) & (1u << core)) != 0u;
        if (m_autoDma[core].empty())
        {
            if (m_trace && enabled)
            {
                ++m_autoDmaUnderflowFrames[core];
                if (!m_autoDmaUnderflowActive[core])
                {
                    ++m_autoDmaUnderflowRuns[core];
                    m_autoDmaUnderflowActive[core] = true;
                }
            }
            return {};
        }

        if (m_trace)
        {
            const size_t depth = m_autoDma[core].size();
            m_autoDmaMinimumDepth[core] =
                std::min(m_autoDmaMinimumDepth[core], depth);
            m_autoDmaMaximumDepth[core] =
                std::max(m_autoDmaMaximumDepth[core], depth);
            m_autoDmaUnderflowActive[core] = false;
        }
        const StereoFrame frame = m_autoDma[core].front();
        m_autoDma[core].pop_front();
        return frame;
    }

    void IopSpu2::traceRenderedSample()
    {
        if (!m_trace || ++m_traceFrames < 48000u)
            return;

        std::fprintf(
            stderr,
            "[ps2xIOP][audio] rendered_frames=%llu "
            "autodma0={enabled=%u,min=%zu,max=%zu,underflow_frames=%llu,underflow_runs=%llu} "
            "autodma1={enabled=%u,min=%zu,max=%zu,underflow_frames=%llu,underflow_runs=%llu}\n",
            static_cast<unsigned long long>(m_traceFrames),
            (reg(0u, kAutoDma) & 0x1u) != 0u ? 1u : 0u,
            m_autoDmaMinimumDepth[0] == std::numeric_limits<size_t>::max()
                ? 0u
                : m_autoDmaMinimumDepth[0],
            m_autoDmaMaximumDepth[0],
            static_cast<unsigned long long>(m_autoDmaUnderflowFrames[0]),
            static_cast<unsigned long long>(m_autoDmaUnderflowRuns[0]),
            (reg(1u, kAutoDma) & 0x2u) != 0u ? 1u : 0u,
            m_autoDmaMinimumDepth[1] == std::numeric_limits<size_t>::max()
                ? 0u
                : m_autoDmaMinimumDepth[1],
            m_autoDmaMaximumDepth[1],
            static_cast<unsigned long long>(m_autoDmaUnderflowFrames[1]),
            static_cast<unsigned long long>(m_autoDmaUnderflowRuns[1]));

        m_traceFrames %= 48000u;
        m_autoDmaUnderflowFrames = {};
        m_autoDmaUnderflowRuns = {};
        m_autoDmaMinimumDepth.fill(std::numeric_limits<size_t>::max());
        m_autoDmaMaximumDepth = {};
    }

    IopSpu2::StereoFrame IopSpu2::renderSample()
    {
        int64_t outputLeft = 0;
        int64_t outputRight = 0;
        for (uint32_t core = 0u; core < kCores; ++core)
        {
            int64_t coreLeft = 0;
            int64_t coreRight = 0;
            for (uint32_t index = 0u; index < kVoices; ++index)
            {
                const int32_t sample = renderVoice(core, index);
                Voice &voice = m_voices[core][index];
                const uint32_t registers = index * kVoiceStride;
                coreLeft +=
                    (static_cast<int64_t>(sample) *
                     voiceVolume(reg(core, registers + kVolumeLeft),
                                 voice.volumeEnvelopes[0])) >>
                    15u;
                coreRight +=
                    (static_cast<int64_t>(sample) *
                     voiceVolume(reg(core, registers + kVolumeRight),
                                 voice.volumeEnvelopes[1])) >>
                    15u;
            }

            const StereoFrame input = renderAutoDma(core);
            coreLeft +=
                (static_cast<int64_t>(input.left) *
                 static_cast<int16_t>(m_inputVolume[core][0])) >>
                15u;
            coreRight +=
                (static_cast<int64_t>(input.right) *
                 static_cast<int16_t>(m_inputVolume[core][1])) >>
                15u;

            outputLeft +=
                (coreLeft * masterVolume(m_masterVolume[core][0])) >> 15u;
            outputRight +=
                (coreRight * masterVolume(m_masterVolume[core][1])) >> 15u;
        }
        return {clampSample(outputLeft), clampSample(outputRight)};
    }

    void IopSpu2::advance()
    {
        const uint64_t now = sampleNow();
        if (now < m_renderedSample)
        {
            m_renderedSample = now;
            return;
        }
        if (!m_audioStarted)
        {
            m_renderedSample = now;
            return;
        }

        const uint64_t frames = now - m_renderedSample;
        if (frames > 0u && frames <= std::numeric_limits<size_t>::max() / 2u)
            m_audio.reserve(m_audio.size() + static_cast<size_t>(frames) * 2u);
        while (m_renderedSample < now)
        {
            const StereoFrame frame = renderSample();
            m_audio.push_back(frame.left);
            m_audio.push_back(frame.right);
            traceRenderedSample();
            ++m_renderedSample;
        }
    }

    std::vector<int16_t> IopSpu2::takeAudio(size_t minimumFrames)
    {
        std::vector<int16_t> result;
        if (!takeAudio(result, minimumFrames))
            return {};
        return result;
    }

    bool IopSpu2::takeAudio(std::vector<int16_t> &destination,
                            size_t minimumFrames)
    {
        if (m_audio.size() / 2u < minimumFrames)
            return false;
        destination.clear();
        destination.swap(m_audio);
        m_audio.reserve(4096u);
        return true;
    }

    void IopSpu2::keyOn(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        const uint32_t registers = index * kVoiceStride;
        const uint32_t addresses = kAddresses + index * kAddressStride;
        voice = {};
        voice.decodeAddress = address(core, addresses + kStart) & ~(kBlockWords - 1u);
        voice.playing = true;
        voice.envelopePhase = EnvelopePhase::Attack;
        if ((reg(core, registers + kVolumeLeft) & 0x8000u) == 0u)
            voice.volumeEnvelopes[0].level = fixedVolume(reg(core, registers + kVolumeLeft));
        if ((reg(core, registers + kVolumeRight) & 0x8000u) == 0u)
            voice.volumeEnvelopes[1].level = fixedVolume(reg(core, registers + kVolumeRight));

        decodeBlock(core, index, 0u);
        decodeBlock(core, index, 1u);
        reg(core, registers + kEnvx) = 0u;
        reg(core, kEndx + (index >= 16u ? 2u : 0u)) &=
            static_cast<uint16_t>(~(1u << (index & 15u)));
        updateNextAddress(core, index);
        m_audioStarted = true;
    }

    void IopSpu2::keyOff(uint32_t core, uint32_t index)
    {
        Voice &voice = m_voices[core][index];
        if (voice.playing)
            voice.envelopePhase = EnvelopePhase::Release;
    }

    uint16_t IopSpu2::read16(uint32_t physical)
    {
        advance();
        const uint32_t relative = (physical - Base) & ~1u;
        if (relative >= kMasterBase && relative < kMasterEnd)
        {
            const uint32_t core = (relative - kMasterBase) / kMasterStride;
            const uint32_t offset = (relative - kMasterBase) % kMasterStride;
            if (offset == kMasterVolumeLeft || offset == kMasterVolumeRight)
                return m_masterVolume[core][offset / 2u];
            if (offset == kInputVolumeLeft || offset == kInputVolumeRight)
                return m_inputVolume[core][(offset - kInputVolumeLeft) / 2u];
        }

        const uint32_t core = relative / kCoreSize;
        const uint32_t offset = relative % kCoreSize;
        if (offset < kVoices * kVoiceStride)
        {
            const uint32_t field = offset % kVoiceStride;
            if (field == kCurrentVolumeLeft || field == kCurrentVolumeRight)
            {
                return static_cast<uint16_t>(
                    m_voices[core][offset / kVoiceStride]
                        .volumeEnvelopes[(field - kCurrentVolumeLeft) / 2u]
                        .level);
            }
        }
        return reg(core, offset);
    }

    void IopSpu2::write16(uint32_t physical, uint16_t value)
    {
        advance();
        const uint32_t relative = (physical - Base) & ~1u;
        if (relative >= kMasterBase && relative < kMasterEnd)
        {
            m_regs[relative / 2u] = value;
            const uint32_t core = (relative - kMasterBase) / kMasterStride;
            const uint32_t offset = (relative - kMasterBase) % kMasterStride;
            if (offset == kMasterVolumeLeft || offset == kMasterVolumeRight)
                m_masterVolume[core][offset / 2u] = value;
            else if (offset == kInputVolumeLeft || offset == kInputVolumeRight)
                m_inputVolume[core][(offset - kInputVolumeLeft) / 2u] = value;
            return;
        }

        const uint32_t core = relative / kCoreSize;
        const uint32_t offset = relative % kCoreSize;
        if (offset < kVoices * kVoiceStride)
        {
            const uint32_t index = offset / kVoiceStride;
            const uint32_t field = offset % kVoiceStride;
            if (field == kEnvx)
            {
                m_voices[core][index].envelope.level = value;
            }
            else if (field == kVolumeLeft || field == kVolumeRight)
            {
                Envelope &envelope =
                    m_voices[core][index].volumeEnvelopes[field / 2u];
                if ((value & 0x8000u) == 0u)
                    envelope.level = fixedVolume(value);
            }
        }
        else if (offset >= kAddresses && offset < kAddressesEnd)
        {
            const uint32_t index = (offset - kAddresses) / kAddressStride;
            const uint32_t field = (offset - kAddresses) % kAddressStride;
            if (field >= kNext)
                return;
            if (field >= kLoop)
                m_voices[core][index].explicitLoop = true;
        }
        else if (offset == kKeyOn || offset == kKeyOn + 2u ||
                 offset == kKeyOff || offset == kKeyOff + 2u)
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
            reg(core, offset) = 0u;
            return;
        }
        else if (offset == kTransferData)
        {
            const uint32_t target = address(core, kTransferAddress);
            std::memcpy(m_ram.data() + target * 2u, &value, sizeof(value));
            setAddress(core, kTransferAddress, target + 1u);
            return;
        }
        else if (offset == kAutoDma && (value & (1u << core)) == 0u)
        {
            m_autoDma[core].clear();
        }
        reg(core, offset) = value;
    }

    void IopSpu2::writeAutoDma(uint32_t core, std::span<const uint8_t> bytes)
    {
        for (size_t offset = 0u; offset < bytes.size();)
        {
            const size_t blockBytes =
                std::min(kAutoDmaBlockBytes, bytes.size() - offset);
            const size_t channelBytes = blockBytes / 2u;
            const size_t frames = channelBytes / sizeof(int16_t);
            for (size_t frame = 0u; frame < frames; ++frame)
            {
                int16_t left = 0;
                int16_t right = 0;
                std::memcpy(&left,
                            bytes.data() + offset + frame * sizeof(int16_t),
                            sizeof(left));
                std::memcpy(&right,
                            bytes.data() + offset + channelBytes +
                                frame * sizeof(int16_t),
                            sizeof(right));
                m_autoDma[core].push_back({left, right});
            }
            offset += blockBytes;
        }
        if (!bytes.empty())
            m_audioStarted = true;
    }

    void IopSpu2::dmaWrite(uint32_t core, std::span<const uint8_t> bytes)
    {
        advance();
        if ((reg(core, kAutoDma) & (1u << core)) != 0u)
        {
            writeAutoDma(core, bytes);
            return;
        }

        uint32_t target = address(core, kTransferAddress) * 2u;
        for (size_t done = 0u; done < bytes.size();)
        {
            const size_t count =
                std::min<size_t>(bytes.size() - done, RamBytes - target);
            std::memcpy(m_ram.data() + target, bytes.data() + done, count);
            done += count;
            target = static_cast<uint32_t>(target + count) & kByteMask;
        }
        setAddress(core, kTransferAddress, target / 2u);
    }

    void IopSpu2::completeDma(uint32_t core)
    {
        reg(core, kStatx) |= kStatxDmaReady;
    }
}
