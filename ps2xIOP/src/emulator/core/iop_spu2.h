#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

namespace ps2x::iop::detail
{
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
        void advance();
        [[nodiscard]] std::vector<int16_t> takeAudio(size_t minimumFrames = 0u);
        bool takeAudio(std::vector<int16_t> &destination, size_t minimumFrames);

    private:
        static constexpr uint32_t kCores = 2u;
        static constexpr uint32_t kVoices = 24u;
        static constexpr uint32_t kDecodedSamples = 56u;

        enum class EnvelopePhase : uint8_t
        {
            Off,
            Attack,
            Decay,
            Sustain,
            Release,
        };

        struct Envelope
        {
            uint32_t counter = 0u;
            int32_t level = 0;
        };

        struct Voice
        {
            std::array<int16_t, kDecodedSamples> decoded{};
            std::array<uint32_t, 2> blocks{};
            std::array<uint8_t, 2> flags{};
            std::array<Envelope, 2> volumeEnvelopes{};
            Envelope envelope{};
            int32_t history1 = 0;
            int32_t history2 = 0;
            uint32_t decodeAddress = 0;
            uint32_t phase = 0;
            EnvelopePhase envelopePhase = EnvelopePhase::Off;
            bool playing = false;
            bool explicitLoop = false;
        };

        struct StereoFrame
        {
            int16_t left = 0;
            int16_t right = 0;
        };

        [[nodiscard]] uint64_t sampleNow() const noexcept;
        [[nodiscard]] uint16_t &reg(uint32_t core, uint32_t offset);
        [[nodiscard]] uint32_t address(uint32_t core, uint32_t offset);
        void setAddress(uint32_t core, uint32_t offset, uint32_t value);
        void decodeBlock(uint32_t core, uint32_t index, uint32_t slot);
        void stepEnvelope(Envelope &envelope, uint32_t rate, bool decrease, bool exponential);
        void updateEnvelope(uint32_t core, uint32_t index);
        [[nodiscard]] int32_t voiceVolume(uint16_t value, Envelope &envelope);
        [[nodiscard]] int32_t renderVoice(uint32_t core, uint32_t index);
        [[nodiscard]] StereoFrame renderAutoDma(uint32_t core);
        [[nodiscard]] StereoFrame renderSample();
        void updateNextAddress(uint32_t core, uint32_t index);
        void keyOn(uint32_t core, uint32_t index);
        void keyOff(uint32_t core, uint32_t index);
        void writeAutoDma(uint32_t core, std::span<const uint8_t> bytes);
        void traceRenderedSample();

        std::vector<uint8_t> m_ram;
        std::array<uint16_t, Size / 2u> m_regs{};
        std::array<std::array<Voice, kVoices>, kCores> m_voices{};
        std::array<std::deque<StereoFrame>, kCores> m_autoDma{};
        std::array<std::array<uint16_t, 2>, kCores> m_masterVolume{};
        std::array<std::array<uint16_t, 2>, kCores> m_inputVolume{};
        std::vector<int16_t> m_audio;
        uint64_t m_renderedSample = 0u;
        bool m_audioStarted = false;
        bool m_trace = false;
        uint64_t m_traceFrames = 0u;
        std::array<uint64_t, kCores> m_autoDmaUnderflowFrames{};
        std::array<uint64_t, kCores> m_autoDmaUnderflowRuns{};
        std::array<size_t, kCores> m_autoDmaMinimumDepth{};
        std::array<size_t, kCores> m_autoDmaMaximumDepth{};
        std::array<bool, kCores> m_autoDmaUnderflowActive{};
        const uint64_t *m_clock = nullptr;
    };
}
