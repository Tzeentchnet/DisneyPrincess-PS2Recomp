#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace ps2x::iop::detail
{
    // The part of the SPU2 that IOP sound drivers can observe: sample RAM,
    // transfers, and each voice's cursor, ENDX bit and ENVX. Drivers stream by
    // refilling behind NAX and free voices by polling ENVX and ENDX, so these
    // have to advance even though nothing is mixed here.
    class IopSpu2
    {
    public:
        static constexpr uint32_t Base = 0x1F900000u;
        static constexpr uint32_t Size = 0x800u;
        static constexpr uint32_t RamBytes = 2u * 1024u * 1024u;
        static constexpr uint64_t IopCyclesPerSample = 768u; // 36.864 MHz / 48 kHz

        IopSpu2();

        void reset();
        void setClock(const uint64_t *iopCycles) noexcept { m_clock = iopCycles; }

        [[nodiscard]] static bool contains(uint32_t physical) noexcept
        {
            return physical - Base < Size;
        }

        [[nodiscard]] uint16_t read16(uint32_t physical);
        void write16(uint32_t physical, uint16_t value);

        // A DMA block into sample RAM at the core's transfer address.
        void dmaWrite(uint32_t core, std::span<const uint8_t> bytes);
        void completeDma(uint32_t core);

    private:
        static constexpr uint32_t kCores = 2u;
        static constexpr uint32_t kVoices = 24u;

        struct Voice
        {
            uint32_t block = 0;        // word address of the ADPCM block being played
            uint32_t phase = 0;        // position in that block, in 1/4096 samples
            uint64_t sample = 0;       // output sample this state is current at
            bool playing = false;
            bool explicitLoop = false; // LSAX written since key on
        };

        [[nodiscard]] uint64_t sampleNow() const noexcept;
        [[nodiscard]] uint16_t &reg(uint32_t core, uint32_t offset);
        [[nodiscard]] uint32_t address(uint32_t core, uint32_t offset);
        void setAddress(uint32_t core, uint32_t offset, uint32_t value);
        void enterBlock(uint32_t core, uint32_t index);
        void advance(uint32_t core, uint32_t index);
        void advanceCore(uint32_t core);
        void keyOn(uint32_t core, uint32_t index);
        void keyOff(uint32_t core, uint32_t index);

        std::vector<uint8_t> m_ram;
        std::array<uint16_t, Size / 2u> m_regs{};
        std::array<std::array<Voice, kVoices>, kCores> m_voices{};
        const uint64_t *m_clock = nullptr;
    };
}
