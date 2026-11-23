#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_parallel_backend.h"
#include "runtime/ps2_memory.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    void check(bool ok, const char *message)
    {
        if (!ok)
            throw std::runtime_error(message);
    }
    uint32_t readFramePixel(const std::shared_ptr<GSGpuFrame> &frame,
                            uint32_t width, uint32_t height,
                            uint32_t x, uint32_t y)
    {
        std::vector<uint8_t> pixels;
        check(frame && frame->GetApi() == GSGpuFrame::Api::Vulkan &&
                  frame->Readback(pixels),
              "Vulkan frame readback failed");
        const size_t offset = (size_t(y) * width + x) * sizeof(uint32_t);
        check(x < width && y < height && offset + sizeof(uint32_t) <= pixels.size(),
              "Vulkan frame readback bounds invalid");
        uint32_t color = 0;
        std::memcpy(&color, pixels.data() + offset, sizeof(color));
        return color;
    }
    void setupTransfer(GS &gs, unsigned psm, unsigned width, unsigned height, unsigned direction, unsigned src = 0, unsigned dst = 0)
    {
        gs.writeRegister(GS_REG_BITBLTBUF, uint64_t(src) | (1ull << 16) | (uint64_t(psm) << 24) |
                                               (uint64_t(dst) << 32) | (1ull << 48) | (uint64_t(psm) << 56));
        gs.writeRegister(GS_REG_TRXPOS, 0);
        gs.writeRegister(GS_REG_TRXREG, width | (uint64_t(height) << 32));
        gs.writeRegister(GS_REG_TRXDIR, direction);
    }
    void testTransfer(GS &gs, unsigned psm, unsigned bpp)
    {
        constexpr unsigned count = 13;
        const auto bytes = (count * bpp + 7) / 8;
        std::vector<uint8_t> input(bytes), output(bytes);
        for (unsigned i = 0; i < bytes; ++i)
            input[i] = uint8_t(31 + i * 7);
        if (bpp == 4)
            input.back() &= 15;
        uint64_t bitblt = (1ull << 16) | (uint64_t(psm) << 24) | (1ull << 48) | (uint64_t(psm) << 56);
        // Exercise the native upload and its final incomplete HWREG word.
        gs.uploadImageNative(bitblt, 0, count | (1ull << 32), 0, input.data(), unsigned(input.size()));
        if (bpp == 4)
        {
            // GS FIFO does not support 4-bit readback; check these uploads/copies in VRAM.
            for (unsigned base : {0u, 64u})
            {
                if (base)
                    setupTransfer(gs, psm, count, 1, 2, 0, base);
                for (unsigned x = 0; x < count; ++x)
                    check(gs.ReadVram(psm, base, 1, x, 0) == ((input[x / 2] >> ((x & 1) * 4)) & 15), "4-bit transfer mismatch");
            }
            std::cout << "PSM " << psm << ": indexed upload and local copy passed\n";
            return;
        }
        setupTransfer(gs, psm, count, 1, 1);
        check(gs.consumeLocalToHostBytes(output.data(), 3) == 3, "short FIFO read failed");
        check(gs.consumeLocalToHostBytes(output.data() + 3, bytes - 3) == bytes - 3, "FIFO tail lost");
        if (input != output)
            throw std::runtime_error("upload/FIFO mismatch for PSM " + std::to_string(psm));
        check(gs.consumeLocalToHostBytes(output.data(), 1) == 0, "FIFO exceeded transfer bounds");
        setupTransfer(gs, psm, count, 1, 2, 0, 64);
        setupTransfer(gs, psm, count, 1, 1, 64);
        check(gs.consumeLocalToHostBytes(output.data(), bytes) == bytes && input == output, "local copy mismatch");
        std::cout << "PSM " << psm << ": native upload, partial FIFO, local copy passed\n";
    }
    void sprite(GS &gs, uint32_t color, uint32_t depth = 0, bool textured = false)
    {
        gs.writeRegister(GS_REG_PRIM, GS_PRIM_SPRITE | (textured ? 0x110 : 0));
        gs.writeRegister(GS_REG_RGBAQ, color | (uint64_t(0x3f800000) << 32));
        gs.writeRegister(GS_REG_UV, 0);
        gs.writeRegister(GS_REG_XYZ2, uint64_t(depth) << 32);
        gs.writeRegister(GS_REG_UV, (2ull * 16) | ((2ull * 16) << 16));
        gs.writeRegister(GS_REG_XYZ2, (8ull * 16) | ((8ull * 16) << 16) | (uint64_t(depth) << 32));
    }
    void testRaster(GS &gs)
    {
        gs.writeRegister(GS_REG_FRAME_1, 1ull << 16);
        gs.writeRegister(GS_REG_SCISSOR_1, (63ull << 16) | (63ull << 48));
        gs.writeRegister(GS_REG_XYOFFSET_1, 0);
        gs.writeRegister(GS_REG_ZBUF_1, 32 | (1ull << 32));
        gs.writeRegister(GS_REG_TEST_1, 1ull << 16 | 1ull << 17);
        gs.writeRegister(GS_REG_PRMODECONT, 1);
        sprite(gs, 0x80402010);
        check(gs.ReadVram(GS_PSM_CT32, 0, 1, 2, 2) == 0x80402010, "solid sprite raster failed");
        gs.writeRegister(GS_REG_ZBUF_1, 32);
        gs.writeRegister(GS_REG_TEST_1, 1ull << 16 | 2ull << 17); // GEQUAL
        gs.WriteVram(GS_PSM_Z32, 32 * 32, 1, 2, 2, 100);
        sprite(gs, 0x80ffffff, 99);
        check(gs.ReadVram(GS_PSM_CT32, 0, 1, 2, 2) == 0x80402010, "depth rejection failed");
        sprite(gs, 0x80a0b0c0, 101);
        check(gs.ReadVram(GS_PSM_CT32, 0, 1, 2, 2) == 0x80a0b0c0, "depth acceptance failed");
        gs.writeRegister(GS_REG_TEST_1, 1ull << 16 | 1ull << 17);
        gs.writeRegister(GS_REG_ZBUF_1, 32 | (1ull << 32));
        for (unsigned y = 0; y < 2; ++y)
            for (unsigned x = 0; x < 2; ++x)
                gs.WriteVram(GS_PSM_T8, 256, 1, x, y, 1);
        gs.WriteVram(GS_PSM_CT32, 512, 1, 1, 0, 0x80224466);
        gs.writeRegister(GS_REG_TEXCLUT, 1);
        gs.writeRegister(GS_REG_TEX0_1, 256 | (1ull << 14) | (uint64_t(GS_PSM_T8) << 20) |
                                            (1ull << 26) | (1ull << 30) | (1ull << 34) | (1ull << 35) |
                                            (512ull << 37) | (1ull << 55) | (1ull << 61));
        sprite(gs, 0x80808080, 0, true);
        check(gs.ReadVram(GS_PSM_CT32, 0, 1, 2, 2) == 0x80224466, "indexed texture/CLUT failed");
        // Fixed-factor blending: (Cs - Cd) * FIX / 128 + Cd, FIX = 64.
        gs.writeRegister(GS_REG_ALPHA_1, (1ull << 2) | (2ull << 4) | (1ull << 6) | (64ull << 32));
        gs.writeRegister(GS_REG_PRIM, GS_PRIM_SPRITE | 64);
        gs.writeRegister(GS_REG_RGBAQ, 0x806688aa);
        gs.writeRegister(GS_REG_XYZ2, 0);
        gs.writeRegister(GS_REG_XYZ2, 128 | (128ull << 16));
        check(gs.ReadVram(GS_PSM_CT32, 0, 1, 2, 2) == 0x80446688, "blending failed");
        gs.writeRegister(GS_REG_SIGNAL, (0xffffffffull << 32) | 123);
        gs.writeRegister(GS_REG_LABEL, (0xffffffffull << 32) | 456);
        gs.writeRegister(GS_REG_FINISH, 0);
        std::cout << "Raster, depth, CLUT/indexed texture, blending and FINISH passed\n";
    }
    void testGIF(GS &gs)
    {
        const uint64_t tag[] = {1ull | (1ull << 15) | (1ull << 60), 14};
        const uint64_t one[] = {17ull << 16, GS_REG_FRAME_1};
        const uint64_t two[] = {23ull << 16, GS_REG_FRAME_2};
        // Incomplete tags on independent paths must survive interleaving.
        gs.processGIFPacket(reinterpret_cast<const uint8_t *>(tag), 16, 1);
        gs.processGIFPacket(reinterpret_cast<const uint8_t *>(tag), 16, 2);
        gs.processGIFPacket(reinterpret_cast<const uint8_t *>(two), 16, 2);
        gs.processGIFPacket(reinterpret_cast<const uint8_t *>(one), 16, 1);
        auto snapshot = gs.getDebugSnapshot();
        check(snapshot.ctx[0].frame.fbw == 17 && snapshot.ctx[1].frame.fbw == 23, "interleaved GIF paths corrupted register state");

        uint64_t packet[] = {
            4ull | (1ull << 60),
            14,
            (1ull << 48) | (128ull << 32),
            GS_REG_BITBLTBUF,
            0,
            GS_REG_TRXPOS,
            3ull | (1ull << 32),
            GS_REG_TRXREG,
            0,
            GS_REG_TRXDIR,
            1ull | (1ull << 15) | (2ull << 58),
            0,
            0x1020304050607080ull,
            0x90a0b0c0ull,
        };
        gs.processGIFPacket(reinterpret_cast<const uint8_t *>(packet), sizeof(packet), 3);
        check(gs.ReadVram(0, 128, 1, 0, 0) == 0x50607080 &&
                  gs.ReadVram(0, 128, 1, 1, 0) == 0x10203040 &&
                  gs.ReadVram(0, 128, 1, 2, 0) == 0x90a0b0c0,
              "GIF IMAGE differs from native upload");
        std::cout << "Interleaved PATH1/2 and GIF IMAGE transfer passed\n";
    }
    void testPackedRaster(GS &gs)
    {
        gs.writeRegister(GS_REG_FRAME_1, 1ull << 16);
        gs.writeRegister(GS_REG_SCISSOR_1, (63ull << 16) | (63ull << 48));
        gs.writeRegister(GS_REG_XYOFFSET_1, 0);
        gs.writeRegister(GS_REG_ZBUF_1, 32 | (1ull << 32));
        gs.writeRegister(GS_REG_TEST_1, 1ull << 16 | 1ull << 17);
        gs.writeRegister(GS_REG_PRMODECONT, 1);
        gs.WriteVram(GS_PSM_CT32, 0, 1, 2, 2, 0);

        const uint64_t packet[] = {
            2ull | (1ull << 15) | (1ull << 46) |
                (uint64_t(GS_PRIM_SPRITE) << 47) | (2ull << 60),
            1ull | (5ull << 4),
            0x00000088000000aaull, 0x0000008000000066ull,
            0, 0,
            0x00000088000000aaull, 0x0000008000000066ull,
            (8ull * 16) | ((8ull * 16) << 32), 0,
        };
        gs.processGIFPacket(reinterpret_cast<const uint8_t *>(packet), sizeof(packet), 1);
        check(gs.ReadVram(GS_PSM_CT32, 0, 1, 2, 2) == 0x806688aa,
              "packed GIF sprite raster failed");
        std::cout << "Packed PATH1 sprite raster passed\n";
    }
    void testPendingPresentation(GS &gs, GSRasterBackend &backend)
    {
        GSPresentationRequest request{};
        request.pmode = 1 | (1ull << 5) | (0xffull << 8);
        request.smode1 = 32ull << 3;
        request.dispfb1 = 10ull << 9;
        request.display1 = 318 | (50ull << 12) | (1ull << 23) | (1279ull << 32) | (447ull << 44);
        std::vector<uint32_t> pixels(640 * 448);
        const auto before = backend.GetReadbackCount();
        for (unsigned pass = 0; pass < 8; ++pass)
        {
            backend.Flush();
            // Force the legal interleaving of a game upload between the frontend's
            // Flush and Present, without relying on thread scheduling or sleeps.
            const uint32_t color = pass & 1 ? 0xff225488 : 0xff6688aa;
            std::fill(pixels.begin(), pixels.end(), color);
            gs.uploadImageNative(10ull << 48, 0, 640ull | (448ull << 32), 0,
                                 reinterpret_cast<const uint8_t *>(pixels.data()),
                                 uint32_t(pixels.size() * sizeof(uint32_t)));
            std::cout << "Present with pending GS upload: " << pass << std::endl;
            auto frame = backend.Present(request);
            check(frame.gpu && frame.width && frame.height, "pending upload produced no scanout");
            const uint32_t center = readFramePixel(
                frame.gpu, frame.width, frame.height, frame.width / 2, frame.height / 2);
            check((center & 0x00ffffffu) == (color & 0x00ffffffu),
                  "pending upload missing from scanout");
        }
        check(backend.GetReadbackCount() == before, "pending presentation requested VRAM readback");
        std::cout << "Uploads between Flush and Present passed\n";
    }
    void testPresentation(GS &gs, GSRegisters &priv)
    {
        // 480p CRTC, only circuit 1 enabled. Use a known color for interop readback.
        gs.writeRegister(GS_REG_FRAME_1, 10ull << 16);
        gs.writeRegister(GS_REG_SCISSOR_1, (639ull << 16) | (447ull << 48));
        check(gs.clearFramebufferContext(0, 0xff225488), "scanout setup clear failed");
        gs.writeRegister(GS_REG_SCISSOR_1, (639ull << 16) | (223ull << 48));
        check(gs.clearFramebufferContext(0, 0xff6688aa), "scanout top-band clear failed");
        gs.writeRegister(GS_REG_FRAME_2, 160 | (10ull << 16));
        gs.writeRegister(GS_REG_SCISSOR_2, (639ull << 16) | (447ull << 48));
        check(gs.clearFramebufferContext(1, 0xff448822), "second circuit clear failed");
        // CRTC ALP uses 0..255, unlike the drawing ALPHA.FIX factor (0x80 = 1).
        priv.pmode = 1 | (1ull << 5) | (0xffull << 8);
        priv.smode1 = 32ull << 3;
        priv.smode2 = 0;
        priv.dispfb1 = 10ull << 9;
        priv.display1 = 318 | (50ull << 12) | (1ull << 23) | (1279ull << 32) | (447ull << 44);
        const auto before = gs.getReadbackCount();
        for (unsigned pass = 0; pass < 6; ++pass)
        {
            if (pass == 2)
            {
                priv.pmode = 3 | (1ull << 5) | (0x80ull << 8);
                priv.dispfb2 = 160 | (10ull << 9);
                priv.display2 = priv.display1;
            }
            if (pass == 3)
            {
                priv.pmode &= ~2ull;
                priv.smode1 |= 2ull << 13;
                priv.smode2 = 3;
                priv.display1 = 636 | (50ull << 12) | (3ull << 23) | (2559ull << 32) | (447ull << 44);
            }
            priv.csr = (pass & 1) << 13;
            priv.vsyncTick++;
            gs.latchHostPresentationFrame();
            uint32_t w = 0, h = 0;
            float aspect = 0;
            auto gpu = gs.getLatchedGpuFrame(w, h, aspect);
            check(gpu && w && h && aspect > 0, "CRTC returned no GPU frame");
            std::vector<uint8_t> pixels;
            check(!gs.copyLatchedHostPresentationFrame(pixels, w, h), "presentation produced CPU pixels");
            gpu = gs.getLatchedGpuFrame(w, h, aspect);
            for (unsigned repeat = 0; repeat < 2; ++repeat)
            {
                if ((pass == 0 || pass == 2) && repeat == 0)
                {
                    const uint32_t center = readFramePixel(gpu, w, h, w / 2, h * 3 / 4);
                    const uint32_t top = readFramePixel(gpu, w, h, w / 2, h / 4);
                    std::cout << "Scanout pass " << pass << ": " << w << 'x' << h
                              << " center=0x" << std::hex << center
                              << " top=0x" << top << std::dec << '\n';
                    if (pass == 0)
                    {
                        check((center & 0x00ffffffu) == 0x00225488u, "Vulkan scanout pixels mismatch");
                        check((top & 0x00ffffffu) == 0x006688aau, "scanout orientation mismatch");
                    }
                    else
                        check((center & 0x00ffffffu) == 0x00336e55u, "dual-circuit composition mismatch");
                }
                check(gpu->GetNativeImage() != nullptr, "repeated Vulkan frame lost its image");
            }
        }
        priv.pmode = 1 | (1ull << 5) | (0xffull << 8);
        priv.smode1 = (32ull << 3) | (2ull << 13);
        priv.smode2 = 1;
        priv.dispfb1 = 10ull << 9;
        priv.display1 = 318 | (50ull << 12) | (1ull << 23) |
                        (1279ull << 32) | (447ull << 44);
        std::vector<uint8_t> fieldPixels[2];
        uint32_t fieldWidth = 0;
        uint32_t fieldHeight = 0;
        for (uint32_t field = 0; field < 2; ++field)
        {
            priv.csr = uint64_t(field) << 13;
            ++priv.vsyncTick;
            gs.latchHostPresentationFrame();
            uint32_t width = 0, height = 0;
            float aspect = 0;
            const auto gpu = gs.getLatchedGpuFrame(width, height, aspect);
            check(gpu && gpu->Readback(fieldPixels[field]),
                  "interlaced field readback failed");
            if (field == 0)
            {
                fieldWidth = width;
                fieldHeight = height;
            }
            else
            {
                check(width == fieldWidth && height == fieldHeight,
                      "interlaced fields changed scanout dimensions");
            }
        }
        check(fieldPixels[0] == fieldPixels[1],
              "alternate field phases produced vertical presentation jitter");
        check(gs.getReadbackCount() == before, "presentation requested VRAM readback");
        std::cout << "Progressive/interlaced Vulkan scanout and repeated frames passed\n";
    }
}
int main()
{
    try
    {
        GS gs;
        GSRegisters priv{};
        std::vector<uint8_t> vram(4 * 1024 * 1024);
        gs.init(vram.data(), uint32_t(vram.size()), &priv);
        auto backend = CreateParallelGSBackend(priv);
        auto *rawBackend = backend.get();
        gs.setRasterBackend(std::move(backend));
        priv.pmode = 3;
        gs.writeRegister(0x59, 32);
        gs.writeRegister(0x59, 32); // Repeated register write is not another flip.
        gs.writeRegister(0x5b, 32);
        check(priv.displayFlipCount[0].load() == 1 && priv.displayFlipCount[1].load() == 1,
              "parallel backend display flips missing or duplicated");
        testPendingPresentation(gs, *rawBackend);
        for (auto [psm, bits] : std::array<std::pair<unsigned, unsigned>, 13>{{{0, 32}, {1, 24}, {2, 16}, {10, 16}, {19, 8}, {20, 4}, {27, 8}, {36, 4}, {44, 4}, {48, 32}, {49, 24}, {50, 16}, {58, 16}}})
            testTransfer(gs, psm, bits);
        testGIF(gs);
        testPackedRaster(gs);
        testRaster(gs);
        check((priv.csr & 3) == 3 && priv.siglblid == (456ull << 32 | 123), "IRQ side effects incorrect");
        testPresentation(gs, priv);
        gs.reset();
        testRaster(gs);
        gs.shutdownBackend();
        std::cout << "Reset and shutdown passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "GPU validation failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
