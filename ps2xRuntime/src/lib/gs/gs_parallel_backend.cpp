#include "runtime/gs/gs_parallel_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "runtime/ps2_memory.h"
#include "ps2_log.h"
#include "context.hpp"
#include "gs_interface.hpp"
#include "thread_id.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace
{
    constexpr size_t VramSize = 4 * 1024 * 1024;

    struct HeadlessDevice
    {
        Vulkan::Context context;
        Vulkan::Device device;

        HeadlessDevice()
        {
            if (!Vulkan::Context::init_loader(nullptr) ||
                !context.init_instance_and_device(nullptr, 0, nullptr, 0))
            {
                throw std::runtime_error("parallel-gs: Vulkan initialization failed");
            }
            device.set_context(context);
            device.init_frame_contexts(2);
            device.next_frame_context();
        }
    };

    class VulkanFrame final : public GSGpuFrame
    {
    public:
        VulkanFrame(Vulkan::Device &device, Vulkan::ImageHandle image)
            : m_device(device), m_image(std::move(image))
        {
        }

        Api GetApi() const override
        {
            return Api::Vulkan;
        }

        const void *GetNativeImage() const override
        {
            return m_image.get();
        }

        bool Readback(std::vector<uint8_t> &pixels) const override
        {
            if (!m_image)
                return false;

            Util::register_thread_index(0);
            Vulkan::BufferCreateInfo info{};
            info.size = size_t(m_image->get_width()) * m_image->get_height() * sizeof(uint32_t);
            info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            info.domain = Vulkan::BufferDomain::CachedHost;
            auto buffer = m_device.create_buffer(info);
            if (!buffer)
                return false;

            auto cmd = m_device.request_command_buffer();
            cmd->copy_image_to_buffer(*buffer, *m_image, 0, {},
                                      {m_image->get_width(), m_image->get_height(), 1},
                                      0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
            cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
            m_device.submit(cmd);
            m_device.wait_idle();

            const auto *mapped = static_cast<const uint8_t *>(
                m_device.map_host_buffer(*buffer, Vulkan::MEMORY_ACCESS_READ_BIT));
            if (!mapped)
                return false;
            pixels.assign(mapped, mapped + info.size);
            m_device.unmap_host_buffer(*buffer, Vulkan::MEMORY_ACCESS_READ_BIT);
            return true;
        }

    private:
        Vulkan::Device &m_device;
        Vulkan::ImageHandle m_image;
    };

    uint64_t word(const void *data)
    {
        uint64_t value;
        std::memcpy(&value, data, 8);
        return value;
    }

    bool profilePresentationEnabled()
    {
        static const bool enabled = []
        {
            const char *value = std::getenv("PS2X_PROFILE_PRESENT");
            return value && value[0] && value[0] != '0';
        }();
        return enabled;
    }

    unsigned bitsPerPixel(unsigned psm)
    {
        switch (psm & 63)
        {
        case GS_PSM_CT24:
        case GS_PSM_Z24:
            return 24;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            return 16;
        case GS_PSM_T8:
        case GS_PSM_T8H:
            return 8;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            return 4;
        default:
            return 32;
        }
    }

    using Reader = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
    using Writer = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    struct PixelAccess
    {
        Reader read;
        Writer write;
    };

    PixelAccess pixelAccess(unsigned psm)
    {
        using namespace GSMem;
        switch (psm & 63)
        {
#define PSM(id, name) \
    case GS_PSM_##id: \
        return {Read##name, Write##name}
            PSM(CT32, CT32);
            PSM(CT24, CT24);
            PSM(CT16, CT16);
            PSM(CT16S, CT16S);
            PSM(T8, P8);
            PSM(T8H, P8H);
            PSM(T4, P4);
            PSM(T4HL, P4HL);
            PSM(T4HH, P4HH);
            PSM(Z32, Z32);
            PSM(Z24, Z24);
            PSM(Z16, Z16);
            PSM(Z16S, Z16S);
#undef PSM
        default:
            return {ReadNull, WriteNull};
        }
    }

    class ParallelBackend final : public GSRasterBackend, private ParallelGS::SignalInterface
    {
        // All GSInterface/Device entry points are serialized, including presentation.
        mutable std::recursive_mutex mutex;
        std::unique_ptr<HeadlessDevice> ownedDevice;
        Vulkan::Device &device;
        mutable ParallelGS::GSInterface gs;
        GSRegisters &priv;
        unsigned threadIndex;
        struct Lock
        {
            std::lock_guard<std::recursive_mutex> guard;
            Lock(std::recursive_mutex &m, unsigned index) : guard(m)
            {
                Util::register_thread_index(index);
            }
        };
        uint64_t bitblt = 0, trxreg = 0;
        GSTransferSnapshot transfer;
        size_t transferBytes = 0, uploadedBytes = 0;
        std::array<uint8_t, 8> uploadTail{};
        size_t uploadTailSize = 0;
        std::array<uint8_t, 16> fifoTail{};
        size_t fifoOffset = 16;
        mutable uint64_t readbacks = 0;
        uint64_t frameWriteIndex = 0;

        void observeRegister(uint8_t addr, uint64_t value,
                             uint32_t path = 0, uint32_t nloop = 0,
                             uint32_t loop = 0, uint32_t reg = 0,
                             uint32_t nreg = 0, uint32_t flg = 0)
        {
            if ((addr == GS_REG_FRAME_1 || addr == GS_REG_FRAME_2) &&
                profilePresentationEnabled())
            {
                const uint32_t fbp = uint32_t(value & 0x1ffu);
                const uint32_t fbw = uint32_t((value >> 16u) & 0x3fu);
                const uint32_t psm = uint32_t((value >> 24u) & 0x3fu);
                const bool validPsm =
                    psm == GS_PSM_CT32 || psm == GS_PSM_CT24 ||
                    psm == GS_PSM_CT16 || psm == GS_PSM_CT16S;
                if (frameWriteIndex < 64u || !validPsm)
                {
                    std::fprintf(
                        stderr,
                        "[parallel-gs-frame-write] index=%llu path=%u reg=0x%02x value=%016llx frame=(%u,%u,%u) tag=(flg=%u,nloop=%u,loop=%u,nreg=%u,reg=%u) valid=%u\n",
                        static_cast<unsigned long long>(frameWriteIndex),
                        path, unsigned(addr),
                        static_cast<unsigned long long>(value),
                        fbp, fbw, psm, flg, nloop, loop, nreg, reg,
                        validPsm ? 1u : 0u);
                    std::fflush(stderr);
                }
                ++frameWriteIndex;
            }
            if (addr == GS_REG_BITBLTBUF)
                bitblt = value;
            if (addr == GS_REG_TRXREG)
                trxreg = value;
            if (addr == GS_REG_TRXDIR)
            {
                transfer = {};
                transfer.direction = value & 3;
                transfer.totalPixels = uint32_t(trxreg & 0xfff) * uint32_t((trxreg >> 32) & 0xfff);
                unsigned psm = unsigned(bitblt >> (transfer.direction == 1 ? 24 : 56)) & 63;
                transferBytes = (size_t(transfer.totalPixels) * bitsPerPixel(psm) + 7) / 8;
                transfer.localToHostPendingBytes = transfer.direction == 1 ? transferBytes : 0;
                if (transfer.direction == 1 && transferBytes)
                    ++readbacks;
                transfer.copiedPixels = transfer.direction == 2 ? transfer.totalPixels : 0;
                uploadedBytes = uploadTailSize = 0;
                fifoOffset = fifoTail.size();
            }

            if (addr == GS_REG_HWREG)
                observeImage(8);
        }
        void observeImage(size_t bytes)
        {
            if (transfer.direction != 0)
                return;
            uploadedBytes = std::min(transferBytes, uploadedBytes + bytes);
            transfer.copiedPixels = std::min<size_t>(transfer.totalPixels, uploadedBytes * 8 / bitsPerPixel(unsigned(bitblt >> 56)));
            auto width = uint32_t(trxreg & 0xfff);
            if (width)
            {
                transfer.x = transfer.copiedPixels % width;
                transfer.y = transfer.copiedPixels / width;
            }
        }

        // Observe transfer boundaries only; the upstream decoder alone executes commands.
        // Start from upstream's saved PATH state, including packets split across DMA calls.
        void observeGIF(uint32_t path, const uint8_t *data, size_t size)
        {
            auto state = gs.get_gif_path(path);
            for (size_t offset = 0; offset < size;)
            {
                if (state.loop == state.tag.NLOOP)
                {
                    std::memcpy(&state.tag, data + offset, 16);
                    state.reg = state.loop = 0;
                    offset += 16;
                    continue;
                }
                unsigned nreg = state.tag.NREG ? state.tag.NREG : 16;
                if (state.tag.FLG >= 2)
                {
                    size_t count = std::min<size_t>(state.tag.NLOOP - state.loop, (size - offset) / 16);
                    observeImage(count * 16);
                    state.loop += uint32_t(count);
                    offset += count * 16;
                }
                else
                {
                    // Only packed A+D can address the transfer registers (>= 0x50).
                    const uint64_t descriptors = word(reinterpret_cast<const uint8_t *>(&state.tag) + 8);
                    if (state.tag.FLG == 0 && ((descriptors >> (state.reg * 4)) & 15) == 14)
                        observeRegister(
                            uint8_t(word(data + offset + 8) & 127),
                            word(data + offset), path, state.tag.NLOOP,
                            state.loop, state.reg, nreg, state.tag.FLG);
                    unsigned count = state.tag.FLG == 0 ? 1 : 2;
                    while (count-- && state.loop < state.tag.NLOOP)
                        if (++state.reg == nreg)
                        {
                            state.reg = 0;
                            ++state.loop;
                        }
                    offset += 16;
                }
            }
        }

        bool on_signal(uint64_t value) override
        {
            auto mask = uint32_t(value >> 32);
            auto id = (uint32_t(priv.siglblid) & ~mask) | (uint32_t(value) & mask);
            priv.siglblid = (priv.siglblid & 0xffffffff00000000ull) | id;
            priv.csr.fetch_or(1);
            return false;
        }

        bool on_finish(uint64_t) override
        {
            // FINISH only orders guest-visible work. VRAM reads synchronise
            // themselves, so stalling the EE until the GPU idles buys nothing.
            gs.flush();
            priv.csr.fetch_or(2);
            return false;
        }

        bool on_label(uint64_t value) override
        {
            auto mask = uint32_t(value >> 32);
            auto id = (uint32_t(priv.siglblid >> 32) & ~mask) | (uint32_t(value) & mask);
            priv.siglblid = (uint64_t(id) << 32) | uint32_t(priv.siglblid);
            return false;
        }

    public:
        ParallelBackend(GSRegisters &registers, Vulkan::Device *externalDevice)
            : ownedDevice(externalDevice ? nullptr : std::make_unique<HeadlessDevice>()),
              device(externalDevice ? *externalDevice : ownedDevice->device),
              priv(registers),
              threadIndex(externalDevice ? 1u : 0u)
        {
            Util::register_thread_index(threadIndex);
            if (!gs.init(&device, ParallelGS::GSOptions{}))
                throw std::runtime_error("parallel-gs: GS initialization failed; check Vulkan feature diagnostics above");
            gs.set_signal_interface(this);
        }

        ~ParallelBackend() override
        {
            Lock lock(mutex, threadIndex);
            gs.flush();
            device.wait_idle();
        }

        bool UsesRawCommands() const override
        {
            return true;
        }

        uint64_t GetReadbackCount() const override
        {
            Lock lock(mutex, threadIndex);
            return readbacks;
        }

        void Initialize(uint8_t *vram, uint32_t size) override
        {
            Lock lock(mutex, threadIndex);
            if (!vram || size < VramSize)
                throw std::invalid_argument("parallel-gs requires 4 MiB initial VRAM");
            std::memcpy(gs.map_vram_write(0, VramSize), vram, VramSize);
            gs.end_vram_write(0, VramSize);
        }

        void Reset() override
        {
            Lock lock(mutex, threadIndex);
            gs.flush();
            device.wait_idle();
            gs.reset_context_state();
            bitblt = trxreg = transferBytes = uploadedBytes = uploadTailSize = 0;
            fifoOffset = 16;
            transfer = {};
        }

        void ProcessGIF(uint32_t path, const uint8_t *data, uint32_t size) override
        {
            Lock lock(mutex, threadIndex);
            if (path < 1 || path > 3 || size % 16)
                throw std::invalid_argument("parallel-gs: invalid GIF path or size");
            if (!data || !size)
                return;
            observeGIF(path, data, size);
            gs.gif_transfer(path, data, size);
        }

        void WriteRegisterRaw(uint8_t addr, uint64_t value) override
        {
            Lock lock(mutex, threadIndex);
            if (addr >= 128)
                return; // Reserved addresses never index upstream's 128-entry table.
            observeRegister(addr, value);
            // Compatibility aliases used by native stubs, not actual GIF registers.
            switch (addr)
            {
            case 0x59:
                priv.writeDisplayFramebuffer(0, value);
                return;
            case 0x5a:
                priv.display1 = value;
                return;
            case 0x5b:
                priv.writeDisplayFramebuffer(1, value);
                return;
            case 0x5c:
                priv.display2 = value;
                return;
            case 0x5f:
                priv.bgcolor = value;
                return;
            }
            gs.write_register(static_cast<ParallelGS::RegisterAddr>(addr), value);
        }

        void Submit(const GSPrimitiveBatch &) override
        {
            throw std::logic_error("parallel-gs received a CPU primitive batch");
        }

        void LoadClut(const GSTex0Reg &, const GSTexClutReg &) override
        {
        }

        void BeginTransfer(const GSTransferCommand &) override
        {
            throw std::logic_error("parallel-gs requires raw transfer registers");
        }

        void UploadImage(const uint8_t *data, uint32_t size) override
        {
            Lock lock(mutex, threadIndex);
            if (!data || transfer.direction != 0)
                return;
            size = uint32_t(std::min<size_t>(size, transferBytes - uploadedBytes));
            observeImage(size);
            while (size)
            {
                auto n = std::min<size_t>(size, 8 - uploadTailSize);
                std::memcpy(uploadTail.data() + uploadTailSize, data, n);
                uploadTailSize += n;
                data += n;
                size -= uint32_t(n);
                if (uploadTailSize == 8 || uploadedBytes == transferBytes && size == 0)
                {
                    std::fill(uploadTail.begin() + uploadTailSize, uploadTail.end(), 0);
                    gs.write_register(ParallelGS::RegisterAddr::HWREG, word(uploadTail.data()));
                    uploadTailSize = 0;
                }
            }
        }

        void Flush() override
        {
            Lock lock(mutex, threadIndex);
            gs.flush();
        }

        void TextureFlush() override
        {
            WriteRegisterRaw(GS_REG_TEXFLUSH, 0);
        }

        void Sync(GSSyncReason reason) override
        {
            Lock lock(mutex, threadIndex);
            if (reason == GSSyncReason::Finish || reason == GSSyncReason::Reset)
            {
                gs.flush();
                device.wait_idle();
            }
            // map_vram_read and the upstream FIFO synchronize their own readbacks.
        }

        PresentationFrame Present(const GSPresentationRequest &request) override
        {
            static uint64_t presentIndex = 0;
            Lock lock(mutex, threadIndex);
            gs.flush();
            const bool samplePresentation =
                presentIndex < 3 || presentIndex == 60 ||
                presentIndex == 180 || presentIndex == 300;
            if (profilePresentationEnabled() && samplePresentation)
            {
                const auto &registers = gs.get_register_state();
                const auto decodeContextFrame = [](uint64_t bits)
                {
                    return GSFrameReg{
                        uint32_t(bits & 0x1ff),
                        uint32_t((bits >> 16) & 0x3f),
                        uint8_t((bits >> 24) & 0x3f),
                        uint32_t(bits >> 32)};
                };
                const auto decodeDisplayFrame = [](uint64_t bits)
                {
                    return GSFrameReg{
                        uint32_t(bits & 0x1ff),
                        uint32_t((bits >> 9) & 0x3f),
                        uint8_t((bits >> 15) & 0x1f),
                        0};
                };
                const GSFrameReg display = decodeDisplayFrame(request.dispfb1);
                const GSFrameReg display2 = decodeDisplayFrame(request.dispfb2);
                const GSFrameReg context0 =
                    decodeContextFrame(registers.ctx[0].frame.bits);
                const GSFrameReg context1 =
                    decodeContextFrame(registers.ctx[1].frame.bits);
                const GSFrameReg &mirror0 = request.contextFrames[0];
                const GSFrameReg &mirror1 = request.contextFrames[1];
                auto *vram = const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(gs.map_vram_read(0, VramSize)));
                size_t nonZeroVramWords = 0;
                for (size_t offset = 0; offset < VramSize; offset += sizeof(uint32_t))
                    nonZeroVramWords += word(vram + offset) != 0;
                const auto stats = gs.consume_flush_stats();
                const auto countNonBlack = [&](const GSFrameReg &frame)
                {
                    size_t count = 0;
                    if (!frame.fbw)
                        return count;
                    auto access = pixelAccess(frame.psm);
                    for (uint32_t y = 0; y < 448; y += 8)
                        for (uint32_t x = 0; x < 640; x += 8)
                            count += (access.read(vram, frame.fbp * 32,
                                                  frame.fbw, x, y) &
                                      0x00ffffffu) != 0;
                    return count;
                };
                struct FrameCandidate
                {
                    size_t nonBlack = 0;
                    uint32_t fbp = 0;
                };
                std::array<FrameCandidate, 8> candidates{};
                for (uint32_t fbp = 0; fbp < 512; ++fbp)
                {
                    const size_t nonBlack = countNonBlack(
                        GSFrameReg{fbp, 8u, GS_PSM_CT32, 0u});
                    if (nonBlack <= candidates.back().nonBlack)
                        continue;
                    auto position = std::find_if(
                        candidates.begin(), candidates.end(),
                        [nonBlack](const FrameCandidate &candidate)
                        {
                            return nonBlack > candidate.nonBlack;
                        });
                    std::move_backward(position, candidates.end() - 1,
                                       candidates.end());
                    *position = {nonBlack, fbp};
                }
                std::fprintf(
                    stderr,
                    "[parallel-gs-frame] present=%llu primitives=%u passes=%u copies=%u vram_words=%zu pmode=%016llx smode1=%016llx smode2=%016llx dispfb1=%016llx display1=%016llx dispfb2=%016llx display2=%016llx preferred=%u preferredSrc=(%u,%u,%u) preferredDest=%u display1=(%u,%u,%u,%zu) display2=(%u,%u,%u,%zu) ctx0=(%u,%u,%u,%zu) ctx1=(%u,%u,%u,%zu) mirror0=(%u,%u,%u) mirror1=(%u,%u,%u) candidates=[%u:%zu,%u:%zu,%u:%zu,%u:%zu,%u:%zu,%u:%zu,%u:%zu,%u:%zu]\n",
                    static_cast<unsigned long long>(presentIndex),
                    stats.num_primitives, stats.num_render_passes,
                    stats.num_copies, nonZeroVramWords,
                    static_cast<unsigned long long>(request.pmode),
                    static_cast<unsigned long long>(request.smode1),
                    static_cast<unsigned long long>(request.smode2),
                    static_cast<unsigned long long>(request.dispfb1),
                    static_cast<unsigned long long>(request.display1),
                    static_cast<unsigned long long>(request.dispfb2),
                    static_cast<unsigned long long>(request.display2),
                    request.hasPreferredSource ? 1u : 0u,
                    request.preferredSource.fbp,
                    request.preferredSource.fbw,
                    unsigned(request.preferredSource.psm),
                    request.preferredDestFbp,
                    display.fbp, display.fbw, display.psm, countNonBlack(display),
                    display2.fbp, display2.fbw, display2.psm, countNonBlack(display2),
                    context0.fbp, context0.fbw, context0.psm, countNonBlack(context0),
                    context1.fbp, context1.fbw, context1.psm, countNonBlack(context1),
                    mirror0.fbp, mirror0.fbw, unsigned(mirror0.psm),
                    mirror1.fbp, mirror1.fbw, unsigned(mirror1.psm),
                    candidates[0].fbp, candidates[0].nonBlack,
                    candidates[1].fbp, candidates[1].nonBlack,
                    candidates[2].fbp, candidates[2].nonBlack,
                    candidates[3].fbp, candidates[3].nonBlack,
                    candidates[4].fbp, candidates[4].nonBlack,
                    candidates[5].fbp, candidates[5].nonBlack,
                    candidates[6].fbp, candidates[6].nonBlack,
                    candidates[7].fbp, candidates[7].nonBlack);
                std::fflush(stderr);
            }
            auto &p = gs.get_priv_register_state();
            p.qwords_lo[0] = request.pmode;
            p.qwords_lo[2] = request.smode1;
            p.qwords_lo[4] = request.smode2;
            p.qwords_lo[12] = request.syncv;
            p.qwords_lo[14] = request.dispfb1;
            p.qwords_lo[16] = request.display1;
            p.qwords_lo[18] = request.dispfb2;
            p.qwords_lo[20] = request.display2;
            p.qwords_lo[28] = request.bgcolor;
            ParallelGS::VSyncInfo info{};
            info.phase = request.field;
            info.dst_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            info.dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            info.dst_access = VK_ACCESS_2_TRANSFER_READ_BIT;
            const bool alternateLineSampling =
                (request.smode2 & 0x1u) != 0u &&
                (request.smode2 & 0x2u) == 0u;
            info.force_progressive = alternateLineSampling;
            info.anti_blur = alternateLineSampling;
            info.crtc_offsets = !alternateLineSampling;
            auto scanout = gs.vsync(info);
            ++presentIndex;
            if (!scanout.image)
                return {};

            PresentationFrame frame;
            frame.width = scanout.image->get_width();
            frame.height = scanout.image->get_height();
            frame.aspectRatio = 4.0f / 3.0f;
            if (scanout.mode_width && scanout.mode_height && scanout.internal_height)
            {
                frame.aspectRatio *=
                    (float(scanout.internal_width) / float(scanout.mode_width)) /
                    (float(scanout.internal_height) / float(scanout.mode_height));
            }
            frame.gpu = std::make_shared<VulkanFrame>(device, std::move(scanout.image));
            return frame;
        }

        uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override
        {
            Lock lock(mutex, threadIndex);
            auto *mapped = const_cast<uint8_t *>(static_cast<const uint8_t *>(gs.map_vram_read(0, VramSize)));
            ++readbacks;
            return pixelAccess(psm).read(mapped, base, bw, x, y);
        }

        void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override
        {
            Lock lock(mutex, threadIndex);
            gs.map_vram_read(0, VramSize); // Preserve untouched pixels and packed-format lanes.
            auto *mapped = static_cast<uint8_t *>(gs.map_vram_write(0, VramSize));
            ++readbacks;
            pixelAccess(psm).write(mapped, base, bw, x, y, value);
            gs.end_vram_write(0, VramSize);
        }

        void SnapshotVram(std::vector<uint8_t> &out) const override
        {
            Lock lock(mutex, threadIndex);
            auto *mapped = static_cast<const uint8_t *>(gs.map_vram_read(0, VramSize));
            ++readbacks;
            out.assign(mapped, mapped + VramSize);
        }

        bool ClearFramebuffer(const GSContext &ctx, uint32_t rgba) override
        {
            Lock lock(mutex, threadIndex);
            if (!ctx.frame.fbw || (ctx.frame.psm != GS_PSM_CT32 && ctx.frame.psm != GS_PSM_CT24 &&
                                   ctx.frame.psm != GS_PSM_CT16 && ctx.frame.psm != GS_PSM_CT16S))
                return false;

            gs.map_vram_read(0, VramSize); // FBMSK and untouched VRAM require current contents.
            auto *mapped = static_cast<uint8_t *>(gs.map_vram_write(0, VramSize));
            ++readbacks;
            auto access = pixelAccess(ctx.frame.psm);

            if (ctx.fba & 1)
                rgba |= 0x80000000u;
            if (bitsPerPixel(ctx.frame.psm) == 16)
                rgba = ((rgba >> 3) & 31) | ((rgba >> 6) & 0x3e0) | ((rgba >> 9) & 0x7c00) | ((rgba >> 16) & 0x8000);

            for (uint32_t y = ctx.scissor.y0; y <= ctx.scissor.y1; ++y)
                for (uint32_t x = ctx.scissor.x0; x <= ctx.scissor.x1; ++x)
                {
                    auto old = access.read(mapped, ctx.frame.fbp * 32, ctx.frame.fbw, x, y);
                    access.write(mapped, ctx.frame.fbp * 32, ctx.frame.fbw, x, y,
                                 (rgba & ~ctx.frame.fbmsk) | (old & ctx.frame.fbmsk));
                }
            gs.end_vram_write(0, VramSize);
            return true;
        }

        uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override
        {
            Lock lock(mutex, threadIndex);
            if (!dst)
                return 0;
            auto count = uint32_t(std::min<size_t>(maxBytes, transfer.localToHostPendingBytes));
            for (uint32_t i = 0; i < count; ++i)
            {
                if (fifoOffset == 16)
                {
                    gs.read_transfer_fifo(fifoTail.data(), 1);
                    fifoOffset = 0;
                }
                dst[i] = fifoTail[fifoOffset++];
            }
            transfer.localToHostPendingBytes -= count;
            return count;
        }

        GSTransferSnapshot GetTransferSnapshot() const override
        {
            Lock lock(mutex, threadIndex);
            return transfer;
        }

        void ReadRegisterState(GSDebugSnapshot &s) const override
        {
            Lock lock(mutex, threadIndex);
            const auto &r = gs.get_register_state();
            for (unsigned i = 0; i < 2; ++i)
            {
                auto &c = s.ctx[i];
                const auto &v = r.ctx[i];
                auto f = v.frame.bits, t = v.tex0.bits, sc = v.scissor.bits, z = v.zbuf.bits;
                c.frame = {uint32_t(f & 511), uint32_t((f >> 16) & 63), uint8_t((f >> 24) & 63), uint32_t(f >> 32)};
                c.zbuf = {uint32_t(z & 511), uint8_t(((z >> 24) & 15) | 0x30), bool((z >> 32) & 1)};
                c.scissor = {uint16_t(sc & 2047), uint16_t((sc >> 16) & 2047), uint16_t((sc >> 32) & 2047), uint16_t((sc >> 48) & 2047)};
                c.xyoffset = {uint16_t(v.xyoffset.bits), uint16_t(v.xyoffset.bits >> 32)};
                c.tex0 = {uint32_t(t & 16383), uint8_t((t >> 14) & 63), uint8_t((t >> 20) & 63), uint8_t((t >> 26) & 15),
                          uint8_t((t >> 30) & 15), uint8_t((t >> 34) & 1), uint8_t((t >> 35) & 3), uint32_t((t >> 37) & 16383),
                          uint8_t((t >> 51) & 15), uint8_t((t >> 55) & 1), uint8_t((t >> 56) & 31), uint8_t(t >> 61)};
                c.tex1 = v.tex1.bits;
                c.miptbp1 = v.miptbl_1_3.bits;
                c.miptbp2 = v.miptbl_4_6.bits;
                c.clamp = v.clamp.bits;
                c.alpha = v.alpha.bits;
                c.test = v.test.bits;
                c.fba = v.fba.bits;
            }

            auto p = r.prim.bits;
            s.prim = {GSPrimType(p & 7), bool(p & 8), bool(p & 16), bool(p & 32), bool(p & 64), bool(p & 128), bool(p & 256), bool(p & 512), bool(p & 1024)};
            auto b = r.bitbltbuf.bits, t = r.trxpos.bits, a = r.texa.bits, c = r.texclut.bits;
            s.texa = {uint8_t(a), bool(a & 0x8000), uint8_t(a >> 32)};
            s.texclut = {uint8_t(c & 63), uint8_t((c >> 6) & 63), uint16_t((c >> 12) & 1023)};
            s.scanmsk = r.scanmsk.bits;
            s.dimx = r.dimx.bits;
            s.dthe = r.dthe.bits;
            s.colclamp = r.colclamp.bits;
            s.bitbltbuf = {uint32_t(b & 16383), uint8_t((b >> 16) & 63), uint8_t((b >> 24) & 63),
                           uint32_t((b >> 32) & 16383), uint8_t((b >> 48) & 63), uint8_t((b >> 56) & 63)};

            s.trxpos = {uint16_t(t & 2047), uint16_t((t >> 16) & 2047), uint16_t((t >> 32) & 2047), uint16_t((t >> 48) & 2047), uint8_t((t >> 59) & 3)};
            s.trxreg = {uint16_t(r.trxreg.bits & 4095), uint16_t((r.trxreg.bits >> 32) & 4095)};
            s.trxdir = uint32_t(r.trxdir.bits & 3);
        }
    };
}

std::unique_ptr<GSRasterBackend> CreateParallelGSBackend(GSRegisters &registers,
                                                         void *vulkanDevice)
{
    return std::make_unique<ParallelBackend>(
        registers, static_cast<Vulkan::Device *>(vulkanDevice));
}
