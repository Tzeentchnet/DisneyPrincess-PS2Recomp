#include "runtime/ps2_audio.h"
#include "runtime/ps2_memory.h"
#include "ps2_host_backend.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
    std::vector<uint8_t> buildWavFromPcm(const int16_t *pcm, size_t sampleCount, uint32_t sampleRate)
    {
        const uint32_t dataSize = static_cast<uint32_t>(sampleCount * 2);
        const uint32_t fileSize = 36 + dataSize;
        std::vector<uint8_t> wav(8 + fileSize);

        uint8_t *p = wav.data();
        p[0] = 'R';
        p[1] = 'I';
        p[2] = 'F';
        p[3] = 'F';
        p[4] = static_cast<uint8_t>(fileSize);
        p[5] = static_cast<uint8_t>(fileSize >> 8);
        p[6] = static_cast<uint8_t>(fileSize >> 16);
        p[7] = static_cast<uint8_t>(fileSize >> 24);
        p[8] = 'W';
        p[9] = 'A';
        p[10] = 'V';
        p[11] = 'E';
        p[12] = 'f';
        p[13] = 'm';
        p[14] = 't';
        p[15] = ' ';
        p[16] = 16;
        p[17] = 0;
        p[18] = 0;
        p[19] = 0;
        p[20] = 1;
        p[21] = 0;
        p[22] = 1;
        p[23] = 0;
        p[24] = static_cast<uint8_t>(sampleRate);
        p[25] = static_cast<uint8_t>(sampleRate >> 8);
        p[26] = static_cast<uint8_t>(sampleRate >> 16);
        p[27] = static_cast<uint8_t>(sampleRate >> 24);
        const uint32_t byteRate = sampleRate * 2;
        p[28] = static_cast<uint8_t>(byteRate);
        p[29] = static_cast<uint8_t>(byteRate >> 8);
        p[30] = static_cast<uint8_t>(byteRate >> 16);
        p[31] = static_cast<uint8_t>(byteRate >> 24);
        p[32] = 2;
        p[33] = 0;
        p[34] = 16;
        p[35] = 0;
        p[36] = 'd';
        p[37] = 'a';
        p[38] = 't';
        p[39] = 'a';
        p[40] = static_cast<uint8_t>(dataSize);
        p[41] = static_cast<uint8_t>(dataSize >> 8);
        p[42] = static_cast<uint8_t>(dataSize >> 16);
        p[43] = static_cast<uint8_t>(dataSize >> 24);
        std::memcpy(p + 44, pcm, dataSize);
        return wav;
    }
}

namespace ps2_vag
{
    bool decode(const uint8_t *data, uint32_t sizeBytes,
                std::vector<int16_t> &outPcm, uint32_t &outSampleRate);
}

struct PS2AudioBackend::Impl
{
    static constexpr size_t StreamFrames = 8192u;
    static constexpr size_t MaxQueuedFrames = 48000u;

    struct TrackedSound
    {
        Sound snd;
        uint32_t sampleKey;
    };

    std::vector<TrackedSound> activeSounds;
#if !defined(PLATFORM_VITA)
    AudioStream stream{};
    bool streamReady = false;
    bool streamPlaying = false;
    size_t primedBuffers = 0u;
#endif
    std::vector<int16_t> queuedAudio;
    size_t queuedAudioRead = 0u;
    bool reportedOverflow = false;
    bool reportedStreamFailure = false;
    bool trace = []
    {
        const char *value = std::getenv("PS2X_TRACE_AUDIO");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    uint64_t tracedFrames = 0u;
    int32_t tracedPeak = 0;
    size_t tracedMinimumQueuedFrames = std::numeric_limits<size_t>::max();
    size_t tracedMaximumQueuedFrames = 0u;
    uint64_t tracedRefillShortages = 0u;
    bool refillShortageActive = false;
    std::atomic<uint64_t> hleCommands{0u};
    std::atomic<uint64_t> hlePlays{0u};
    std::chrono::steady_clock::time_point lastSubmission{};
    uint64_t tracedMaximumSubmissionGapMicros = 0u;
    uint64_t tracedSubmissions = 0u;
    std::chrono::steady_clock::time_point lastDeviceUpdate{};
    uint64_t tracedMaximumDeviceUpdateGapMicros = 0u;
    uint64_t tracedDeviceUpdates = 0u;
};

PS2AudioBackend::PS2AudioBackend() : m_impl(std::make_unique<Impl>())
{
}

PS2AudioBackend::~PS2AudioBackend()
{
    if (m_impl)
        stopAll();
}

void PS2AudioBackend::setAudioReady(bool ready)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_audioReady = ready;
    if (ready)
        return;

#if !defined(PLATFORM_VITA)
    if (m_impl->streamReady)
    {
        if (m_impl->streamPlaying)
            StopAudioStream(m_impl->stream);
        UnloadAudioStream(m_impl->stream);
        m_impl->stream = {};
        m_impl->streamReady = false;
        m_impl->streamPlaying = false;
        m_impl->primedBuffers = 0u;
    }
#endif
    m_impl->queuedAudio.clear();
    m_impl->queuedAudioRead = 0u;
    m_impl->lastSubmission = {};
    m_impl->lastDeviceUpdate = {};
}

void PS2AudioBackend::submitFrames(std::span<const int16_t> interleavedStereo)
{
    if ((interleavedStereo.size() & 1u) != 0u)
    {
        std::cerr << "[ps2xRuntime][audio] rejected an odd interleaved sample count\n";
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_audioReady || interleavedStereo.empty())
        return;

    if (m_impl->trace)
    {
        const auto now = std::chrono::steady_clock::now();
        if (m_impl->lastSubmission.time_since_epoch().count() != 0)
        {
            const uint64_t gapMicros = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    now - m_impl->lastSubmission)
                    .count());
            m_impl->tracedMaximumSubmissionGapMicros =
                std::max(m_impl->tracedMaximumSubmissionGapMicros, gapMicros);
        }
        m_impl->lastSubmission = now;
        ++m_impl->tracedSubmissions;

        for (const int16_t sample : interleavedStereo)
        {
            const int32_t magnitude =
                sample == std::numeric_limits<int16_t>::min()
                    ? 32768
                    : std::abs(static_cast<int32_t>(sample));
            m_impl->tracedPeak = std::max(m_impl->tracedPeak, magnitude);
        }
        m_impl->tracedFrames += interleavedStereo.size() / 2u;
        if (m_impl->tracedFrames >= 48000u)
        {
            const size_t minimumQueuedFrames =
                m_impl->tracedMinimumQueuedFrames == std::numeric_limits<size_t>::max()
                    ? 0u
                    : m_impl->tracedMinimumQueuedFrames;
            const uint64_t hleCommands =
                m_impl->hleCommands.exchange(0u, std::memory_order_relaxed);
            const uint64_t hlePlays =
                m_impl->hlePlays.exchange(0u, std::memory_order_relaxed);
            std::fprintf(
                stderr,
                "[ps2xRuntime][audio] mixed_frames=%llu peak=%d "
                "queued_frames_min=%zu queued_frames_max=%zu "
                "refill_shortages=%llu submissions=%llu "
                "max_submit_gap_ms=%.3f device_updates=%llu "
                "max_device_update_gap_ms=%.3f device_buffer_frames=%zu "
                "hle_commands=%llu hle_plays=%llu\n",
                static_cast<unsigned long long>(m_impl->tracedFrames),
                m_impl->tracedPeak,
                minimumQueuedFrames,
                m_impl->tracedMaximumQueuedFrames,
                static_cast<unsigned long long>(m_impl->tracedRefillShortages),
                static_cast<unsigned long long>(m_impl->tracedSubmissions),
                m_impl->tracedMaximumSubmissionGapMicros / 1000.0,
                static_cast<unsigned long long>(m_impl->tracedDeviceUpdates),
                m_impl->tracedMaximumDeviceUpdateGapMicros / 1000.0,
                Impl::StreamFrames * 2u,
                static_cast<unsigned long long>(hleCommands),
                static_cast<unsigned long long>(hlePlays));
            m_impl->tracedFrames %= 48000u;
            m_impl->tracedPeak = 0;
            m_impl->tracedMinimumQueuedFrames =
                std::numeric_limits<size_t>::max();
            m_impl->tracedMaximumQueuedFrames = 0u;
            m_impl->tracedRefillShortages = 0u;
            m_impl->tracedMaximumSubmissionGapMicros = 0u;
            m_impl->tracedSubmissions = 0u;
            m_impl->tracedMaximumDeviceUpdateGapMicros = 0u;
            m_impl->tracedDeviceUpdates = 0u;
        }
    }

#if defined(PLATFORM_VITA)
    return;
#else
    if (!m_impl->streamReady)
    {
        SetAudioStreamBufferSizeDefault(static_cast<int>(Impl::StreamFrames));
        m_impl->stream = LoadAudioStream(48000u, 16u, 2u);
        m_impl->streamReady = IsAudioStreamValid(m_impl->stream);
        if (!m_impl->streamReady)
        {
            if (!m_impl->reportedStreamFailure)
            {
                std::cerr << "[ps2xRuntime][audio] failed to create the 48 kHz output stream\n";
                m_impl->reportedStreamFailure = true;
            }
            return;
        }
        m_impl->queuedAudio.reserve(Impl::StreamFrames * 8u);
    }

    m_impl->queuedAudio.insert(m_impl->queuedAudio.end(),
                               interleavedStereo.begin(),
                               interleavedStereo.end());
    size_t availableSamples = m_impl->queuedAudio.size() - m_impl->queuedAudioRead;
    const size_t maximumSamples = Impl::MaxQueuedFrames * 2u;
    if (availableSamples > maximumSamples)
    {
        const size_t droppedSamples = (availableSamples - maximumSamples) & ~size_t{1u};
        m_impl->queuedAudioRead += droppedSamples;
        availableSamples -= droppedSamples;
        if (!m_impl->reportedOverflow)
        {
            std::cerr << "[ps2xRuntime][audio] output fell behind; dropping queued frames\n";
            m_impl->reportedOverflow = true;
        }
    }

    std::array<int16_t, Impl::StreamFrames * 2u> block{};
    while (availableSamples >= block.size() &&
           IsAudioStreamProcessed(m_impl->stream))
    {
        std::copy_n(m_impl->queuedAudio.begin() +
                        static_cast<std::ptrdiff_t>(m_impl->queuedAudioRead),
                    block.size(),
                    block.begin());
        UpdateAudioStream(m_impl->stream,
                          block.data(),
                          static_cast<int>(Impl::StreamFrames));
        if (m_impl->trace)
        {
            const auto now = std::chrono::steady_clock::now();
            if (m_impl->lastDeviceUpdate.time_since_epoch().count() != 0)
            {
                const uint64_t gapMicros = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        now - m_impl->lastDeviceUpdate)
                        .count());
                m_impl->tracedMaximumDeviceUpdateGapMicros =
                    std::max(m_impl->tracedMaximumDeviceUpdateGapMicros, gapMicros);
            }
            m_impl->lastDeviceUpdate = now;
            ++m_impl->tracedDeviceUpdates;
        }
        m_impl->queuedAudioRead += block.size();
        availableSamples -= block.size();
        if (!m_impl->streamPlaying)
        {
            ++m_impl->primedBuffers;
            if (m_impl->primedBuffers >= 2u)
            {
                PlayAudioStream(m_impl->stream);
                m_impl->streamPlaying = true;
            }
        }
    }

    if (m_impl->queuedAudioRead == m_impl->queuedAudio.size())
    {
        m_impl->queuedAudio.clear();
        m_impl->queuedAudioRead = 0u;
    }
    else if (m_impl->queuedAudioRead >= Impl::StreamFrames * 8u)
    {
        m_impl->queuedAudio.erase(
            m_impl->queuedAudio.begin(),
            m_impl->queuedAudio.begin() +
                static_cast<std::ptrdiff_t>(m_impl->queuedAudioRead));
        m_impl->queuedAudioRead = 0u;
    }

    if (m_impl->trace)
    {
        const size_t queuedFrames = availableSamples / 2u;
        m_impl->tracedMinimumQueuedFrames =
            std::min(m_impl->tracedMinimumQueuedFrames, queuedFrames);
        m_impl->tracedMaximumQueuedFrames =
            std::max(m_impl->tracedMaximumQueuedFrames, queuedFrames);
        const bool refillShortage =
            m_impl->streamPlaying &&
            IsAudioStreamProcessed(m_impl->stream) &&
            availableSamples < block.size();
        if (refillShortage && !m_impl->refillShortageActive)
        {
            ++m_impl->tracedRefillShortages;
        }
        m_impl->refillShortageActive = refillShortage;
    }
#endif
}

void PS2AudioBackend::onVagTransfer(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes)
{
    if (!rdram || sizeBytes < 48)
        return;

    const uint32_t physAddr = srcAddr & PS2_RAM_MASK;
    if (physAddr + sizeBytes > PS2_RAM_SIZE)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(rdram + physAddr, sizeBytes, pcm, sampleRate))
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = std::move(sample);
    m_mostRecentSampleKey = physAddr;
}

void PS2AudioBackend::onVagTransferFromBuffer(const uint8_t *data, uint32_t sizeBytes, uint32_t keyAddr)
{
    if (!data || sizeBytes < 48)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(data, sizeBytes, pcm, sampleRate))
        return;

    const uint32_t physAddr = keyAddr & PS2_RAM_MASK;
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = sample;
    m_mostRecentSampleKey = physAddr;
    m_loadOrderSamples.push_back(std::move(sample));
    m_loadOrderSampleKeys.push_back(physAddr);
    constexpr size_t kMaxLoadOrderSamples = 32;
    if (m_loadOrderSamples.size() > kMaxLoadOrderSamples)
    {
        m_loadOrderSamples.erase(m_loadOrderSamples.begin());
        m_loadOrderSampleKeys.erase(m_loadOrderSampleKeys.begin());
    }
}

namespace
{
    constexpr uint32_t LIBSD_CMD_SET_VOICE = 0x8010u;
}

void PS2AudioBackend::onSoundCommand(uint32_t sid, uint32_t rpcNum,
                                     const uint8_t *sendBuf, uint32_t sendSize,
                                     uint8_t *recvBuf, uint32_t recvSize)
{
    if (sid != 0x80000701u)
        return;

    if ((rpcNum == LIBSD_CMD_SET_VOICE || (rpcNum & 0xFF00u) == 0x8100u) &&
        sendBuf && sendSize >= 20)
    {
        m_impl->hleCommands.fetch_add(1u, std::memory_order_relaxed);
        uint32_t sampleAddr = 0;
        uint32_t voiceIndex = 0xFFFFFFFFu;
        for (int vo = 4; vo >= 0 && voiceIndex == 0xFFFFFFFFu; vo -= 4)
        {
            if (vo < static_cast<int>(sendSize))
            {
                uint32_t v = 0;
                std::memcpy(&v, sendBuf + vo, sizeof(v));
                if (v < 24u)
                    voiceIndex = v;
            }
        }

        constexpr uint32_t kMinPlausibleAddr = 0x1000u;
        for (int off = 12; off <= 24 && sampleAddr == 0; off += 4)
        {
            if (sendSize >= static_cast<uint32_t>(off + 4))
            {
                uint32_t cand = 0;
                std::memcpy(&cand, sendBuf + off, sizeof(cand));
                if (cand >= kMinPlausibleAddr && (cand <= PS2_RAM_MASK || (cand & ~PS2_RAM_MASK) == 0))
                    sampleAddr = cand;
            }
        }
        if (sampleAddr == 0)
            sampleAddr = m_mostRecentSampleKey;

        float pitch = 1.0f;
        if (sendSize >= 12)
        {
            uint16_t pitchHalf = 0;
            std::memcpy(&pitchHalf, sendBuf + 8, sizeof(pitchHalf));
            if (pitchHalf != 0)
                pitch = 4096.0f / static_cast<float>(pitchHalf);
        }
        play(sampleAddr, pitch, 1.0f, voiceIndex);
    }
}

void PS2AudioBackend::play(uint32_t sampleAddr, float pitch, float volume, uint32_t voiceIndex)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample *sampleToPlay = nullptr;
    uint32_t sampleKey = 0;

    auto it = m_sampleBank.find(sampleAddr & PS2_RAM_MASK);
    if (it != m_sampleBank.end())
    {
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    else if (voiceIndex != 0xFFFFFFFFu &&
             voiceIndex < m_loadOrderSamples.size() &&
             voiceIndex < m_loadOrderSampleKeys.size())
    {
        sampleToPlay = &m_loadOrderSamples[voiceIndex];
        sampleKey = m_loadOrderSampleKeys[voiceIndex];
    }
    else
    {
        it = m_sampleBank.find(m_mostRecentSampleKey);
        if (it == m_sampleBank.end())
            return;
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    if (!sampleToPlay || sampleToPlay->pcm.empty())
        return;

    const bool isBgm = (sampleToPlay->pcm.size() > static_cast<size_t>(sampleToPlay->sampleRate * 5));
    playDecodedSample(sampleKey, *sampleToPlay, pitch, volume, isBgm);
}

void PS2AudioBackend::pruneFinishedSounds()
{
#if defined(PLATFORM_VITA)
    return;
#else
    auto &sounds = m_impl->activeSounds;
    auto it = sounds.begin();
    while (it != sounds.end())
    {
        if (!IsSoundPlaying(it->snd))
        {
            UnloadSound(it->snd);
            it = sounds.erase(it);
        }
        else
        {
            ++it;
        }
    }
#endif
}

void PS2AudioBackend::playDecodedSample(uint32_t sampleKey, DecodedSample &sample, float pitch, float volume,
                                        bool isBgm)
{
#if defined(PLATFORM_VITA)
    (void)sampleKey;
    (void)sample;
    (void)pitch;
    (void)volume;
    (void)isBgm;
    return;
#else
    if (!m_audioReady || sample.pcm.empty())
        return;

    pruneFinishedSounds();

    for (const auto &t : m_impl->activeSounds)
    {
        if (t.sampleKey == sampleKey && IsSoundPlaying(t.snd))
            return;
    }

    auto &sounds = m_impl->activeSounds;
    if (isBgm)
    {
        for (auto it = sounds.begin(); it != sounds.end();)
        {
            if (IsSoundPlaying(it->snd))
            {
                StopSound(it->snd);
                UnloadSound(it->snd);
                it = sounds.erase(it);
            }
            else
                ++it;
        }
    }

    constexpr int kMaxConcurrentSounds = 4;
    while (static_cast<int>(sounds.size()) >= kMaxConcurrentSounds)
    {
        StopSound(sounds.front().snd);
        UnloadSound(sounds.front().snd);
        sounds.erase(sounds.begin());
    }

    std::vector<uint8_t> wav = buildWavFromPcm(sample.pcm.data(), sample.pcm.size(), sample.sampleRate);
    Wave wave = LoadWaveFromMemory(".wav", wav.data(), static_cast<int>(wav.size()));
    if (wave.frameCount <= 0)
        return;
    Sound snd = LoadSoundFromWave(wave);
    UnloadWave(wave);
    SetSoundPitch(snd, pitch);
    SetSoundVolume(snd, volume);
    m_impl->activeSounds.push_back({snd, sampleKey});
    PlaySound(snd);
    m_impl->hlePlays.fetch_add(1u, std::memory_order_relaxed);
#endif
}

void PS2AudioBackend::stop(uint32_t voiceId)
{
    (void)voiceId;
}

void PS2AudioBackend::stopAll()
{
    std::lock_guard<std::mutex> lock(m_mutex);
#if defined(PLATFORM_VITA)
    m_impl->queuedAudio.clear();
    m_impl->queuedAudioRead = 0u;
    return;
#else
    for (auto &t : m_impl->activeSounds)
    {
        StopSound(t.snd);
        UnloadSound(t.snd);
    }
    m_impl->activeSounds.clear();
    if (m_impl->streamReady)
    {
        if (m_impl->streamPlaying)
            StopAudioStream(m_impl->stream);
        UnloadAudioStream(m_impl->stream);
        m_impl->stream = {};
        m_impl->streamReady = false;
        m_impl->streamPlaying = false;
        m_impl->primedBuffers = 0u;
    }
    m_impl->queuedAudio.clear();
    m_impl->queuedAudioRead = 0u;
#endif
}
