#include "ps2_host_backend.h"
#include "runtime/gs/gs_types.h"

#ifdef PS2X_GS_PARALLEL

#include "thread_id.hpp"
#define Rectangle Win32Rectangle
#define CloseWindow Win32CloseWindow
#define ShowCursor Win32ShowCursor
#include "wsi.hpp"
#undef ShowCursor
#undef CloseWindow
#undef Rectangle
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
#include <Xinput.h>

namespace
{
    constexpr wchar_t WindowClassName[] = L"PS2RecompVulkanWindow";

    int virtualKey(PS2HostKey key)
    {
        switch (key)
        {
        case PS2HostKey::Up: return VK_UP;
        case PS2HostKey::Down: return VK_DOWN;
        case PS2HostKey::Left: return VK_LEFT;
        case PS2HostKey::Right: return VK_RIGHT;
        case PS2HostKey::A: return 'A';
        case PS2HostKey::C: return 'C';
        case PS2HostKey::D: return 'D';
        case PS2HostKey::E: return 'E';
        case PS2HostKey::Q: return 'Q';
        case PS2HostKey::S: return 'S';
        case PS2HostKey::V: return 'V';
        case PS2HostKey::W: return 'W';
        case PS2HostKey::X: return 'X';
        case PS2HostKey::Z: return 'Z';
        case PS2HostKey::One: return '1';
        case PS2HostKey::Three: return '3';
        case PS2HostKey::Enter: return VK_RETURN;
        case PS2HostKey::Escape: return VK_ESCAPE;
        case PS2HostKey::Space: return VK_SPACE;
        case PS2HostKey::Tab: return VK_TAB;
        case PS2HostKey::LeftShift: return VK_LSHIFT;
        case PS2HostKey::RightShift: return VK_RSHIFT;
        case PS2HostKey::LeftControl: return VK_LCONTROL;
        case PS2HostKey::RightControl: return VK_RCONTROL;
        case PS2HostKey::Keypad0: return VK_NUMPAD0;
        case PS2HostKey::Keypad1: return VK_NUMPAD1;
        case PS2HostKey::F1: return VK_F1;
        }
        return 0;
    }

    WORD gamepadButton(PS2HostGamepadButton button)
    {
        switch (button)
        {
        case PS2HostGamepadButton::DpadUp: return XINPUT_GAMEPAD_DPAD_UP;
        case PS2HostGamepadButton::DpadDown: return XINPUT_GAMEPAD_DPAD_DOWN;
        case PS2HostGamepadButton::DpadLeft: return XINPUT_GAMEPAD_DPAD_LEFT;
        case PS2HostGamepadButton::DpadRight: return XINPUT_GAMEPAD_DPAD_RIGHT;
        case PS2HostGamepadButton::FaceDown: return XINPUT_GAMEPAD_A;
        case PS2HostGamepadButton::FaceRight: return XINPUT_GAMEPAD_B;
        case PS2HostGamepadButton::FaceLeft: return XINPUT_GAMEPAD_X;
        case PS2HostGamepadButton::FaceUp: return XINPUT_GAMEPAD_Y;
        case PS2HostGamepadButton::LeftShoulder: return XINPUT_GAMEPAD_LEFT_SHOULDER;
        case PS2HostGamepadButton::RightShoulder: return XINPUT_GAMEPAD_RIGHT_SHOULDER;
        case PS2HostGamepadButton::LeftThumb: return XINPUT_GAMEPAD_LEFT_THUMB;
        case PS2HostGamepadButton::RightThumb: return XINPUT_GAMEPAD_RIGHT_THUMB;
        case PS2HostGamepadButton::Select: return XINPUT_GAMEPAD_BACK;
        case PS2HostGamepadButton::Start: return XINPUT_GAMEPAD_START;
        case PS2HostGamepadButton::LeftTrigger:
        case PS2HostGamepadButton::RightTrigger:
            return 0;
        }
        return 0;
    }

    class Win32VulkanPlatform final : public Vulkan::WSIPlatform
    {
    public:
        bool initialize(const char *title, uint32_t clientWidth, uint32_t clientHeight)
        {
            instance = GetModuleHandleW(nullptr);
            WNDCLASSEXW windowClass{};
            windowClass.cbSize = sizeof(windowClass);
            windowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
            windowClass.lpfnWndProc = windowProcedure;
            windowClass.hInstance = instance;
            windowClass.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
            windowClass.lpszClassName = WindowClassName;
            if (!RegisterClassExW(&windowClass) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
                return false;

            RECT bounds{0, 0, static_cast<LONG>(clientWidth), static_cast<LONG>(clientHeight)};
            AdjustWindowRect(&bounds, WS_OVERLAPPEDWINDOW, FALSE);
            const std::wstring wideTitle(title, title + std::strlen(title));
            window = CreateWindowExW(
                0, WindowClassName, wideTitle.c_str(), WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT, CW_USEDEFAULT, bounds.right - bounds.left,
                bounds.bottom - bounds.top, nullptr, nullptr, instance, this);
            if (!window)
                return false;

            width = clientWidth;
            height = clientHeight;
            ShowWindow(window, SW_SHOW);
            UpdateWindow(window);
            return true;
        }

        VkSurfaceKHR create_surface(VkInstance vkInstance, VkPhysicalDevice) override
        {
            VkWin32SurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
            info.hinstance = instance;
            info.hwnd = window;
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            if (vkCreateWin32SurfaceKHR(vkInstance, &info, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            updateDimensions();
            return surface;
        }

        std::vector<const char *> get_instance_extensions() override
        {
            return {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
        }

        uint32_t get_surface_width() override
        {
            updateDimensions();
            return width;
        }

        uint32_t get_surface_height() override
        {
            updateDimensions();
            return height;
        }

        bool alive(Vulkan::WSI &) override
        {
            return !closed;
        }

        void poll_input() override
        {
            MSG message;
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }

        void poll_input_async(Granite::InputTrackerHandler *) override
        {
            poll_input();
        }

        bool isAlive() const
        {
            return !closed;
        }

        bool hasInputFocus() const
        {
            return window && GetForegroundWindow() == window;
        }

        // Keys are sampled when the guest polls, which can be further apart
        // than a tap. A tap therefore counts as held for a few guest frames.
        bool recentlyPressed(int key) const
        {
            constexpr uint64_t kTapHoldMilliseconds = 100;
            const uint64_t pressed = lastPress[key & 0xFF].load(std::memory_order_relaxed);
            return pressed != 0 && GetTickCount64() - pressed < kTapHoldMilliseconds;
        }

        ~Win32VulkanPlatform() override
        {
            if (window)
                DestroyWindow(window);
        }

    private:
        static LRESULT CALLBACK windowProcedure(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
        {
            auto *platform = reinterpret_cast<Win32VulkanPlatform *>(
                GetWindowLongPtrW(hwnd, GWLP_USERDATA));
            if (message == WM_NCCREATE)
            {
                const auto *create = reinterpret_cast<const CREATESTRUCTW *>(lparam);
                platform = static_cast<Win32VulkanPlatform *>(create->lpCreateParams);
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(platform));
            }

            if (platform)
            {
                if (message == WM_CLOSE)
                {
                    platform->closed = true;
                    return 0;
                }
                if (message == WM_DESTROY)
                {
                    platform->closed = true;
                    return 0;
                }
                if (message == WM_KEYDOWN && wparam < 256 && (lparam & (1 << 30)) == 0)
                {
                    int key = static_cast<int>(wparam);
                    // The message reports the generic modifier; polls ask for a side.
                    const bool extended = (lparam & (1 << 24)) != 0;
                    if (key == VK_CONTROL)
                        key = extended ? VK_RCONTROL : VK_LCONTROL;
                    else if (key == VK_SHIFT)
                        key = ((lparam >> 16) & 0xFF) == 0x36 ? VK_RSHIFT : VK_LSHIFT;
                    platform->lastPress[key].store(GetTickCount64(), std::memory_order_relaxed);
                }
                if (message == WM_KILLFOCUS)
                {
                    for (auto &press : platform->lastPress)
                        press.store(0, std::memory_order_relaxed);
                }
                if (message == WM_SIZE && wparam != SIZE_MINIMIZED)
                {
                    platform->width = std::max<uint32_t>(1, LOWORD(lparam));
                    platform->height = std::max<uint32_t>(1, HIWORD(lparam));
                    platform->resize = true;
                }
            }
            return DefWindowProcW(hwnd, message, wparam, lparam);
        }

        void updateDimensions()
        {
            RECT client{};
            if (window && GetClientRect(window, &client))
            {
                width = std::max<LONG>(1, client.right - client.left);
                height = std::max<LONG>(1, client.bottom - client.top);
            }
        }

        HINSTANCE instance = nullptr;
        HWND window = nullptr;
        uint32_t width = 1;
        uint32_t height = 1;
        bool closed = false;
        std::array<std::atomic<uint64_t>, 256> lastPress{};
    };

    struct VulkanHost
    {
        Win32VulkanPlatform platform;
        Vulkan::WSI wsi;

        VulkanHost(const char *title, uint32_t width, uint32_t height)
        {
            if (!platform.initialize(title, width, height))
                throw std::runtime_error("could not create the native Vulkan window");
            if (!Vulkan::Context::init_loader(nullptr))
                throw std::runtime_error("Vulkan loader is unavailable");

            wsi.set_platform(&platform);
            wsi.set_present_mode(Vulkan::PresentMode::SyncToVBlank);
            wsi.set_backbuffer_format(Vulkan::BackbufferFormat::UNORM);
            wsi.set_support_prerotate(false);
            wsi.set_extra_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            Vulkan::Context::SystemHandles handles{};
            if (!wsi.init_simple(2, handles))
                throw std::runtime_error("could not initialize the Vulkan swapchain");
            // parallel-gs advances a frame context on every flush, several times
            // per guest frame. With the default two, each flush waits for the
            // GPU to finish the one before last.
            constexpr unsigned kFrameContexts = 8;
            wsi.get_device().init_frame_contexts(kFrameContexts);
        }

        static void profileFrame(const std::shared_ptr<GSGpuFrame> &frame,
                                 uint32_t frameWidth, uint32_t frameHeight,
                                 float aspectRatio, uint64_t frameIndex)
        {
            static uint64_t validFrameIndex = 0;
            static const bool profileFrames = []
            {
                const char *value = std::getenv("PS2X_PROFILE_PRESENT");
                return value && value[0] && value[0] != '0';
            }();
            if (!profileFrames || !frame ||
                frame->GetApi() != GSGpuFrame::Api::Vulkan ||
                !frame->GetNativeImage() || !frameWidth || !frameHeight)
            {
                return;
            }

            const bool sampleFrame =
                validFrameIndex < 3 || frameIndex == 60 || frameIndex == 300 ||
                frameIndex == 600 || frameIndex == 1200;
            ++validFrameIndex;
            if (!sampleFrame)
                return;

            std::vector<uint8_t> pixels;
            if (frame->Readback(pixels))
            {
                uint64_t hash = 1469598103934665603ull;
                size_t nonBlack = 0;
                for (size_t offset = 0; offset + 3 < pixels.size(); offset += 4)
                {
                    hash ^= pixels[offset + 0];
                    hash *= 1099511628211ull;
                    hash ^= pixels[offset + 1];
                    hash *= 1099511628211ull;
                    hash ^= pixels[offset + 2];
                    hash *= 1099511628211ull;
                    nonBlack += (pixels[offset + 0] | pixels[offset + 1] |
                                 pixels[offset + 2]) != 0;
                }
                uint32_t center = 0;
                const size_t centerOffset =
                    (size_t(frameHeight / 2) * frameWidth + frameWidth / 2) *
                    sizeof(uint32_t);
                if (centerOffset + sizeof(center) <= pixels.size())
                    std::memcpy(&center, pixels.data() + centerOffset, sizeof(center));
                std::fprintf(stderr,
                             "[vulkan-frame] host_frame=%llu size=%ux%u aspect=%.6f nonblack=%zu hash=%016llx center=%08x\n",
                             static_cast<unsigned long long>(frameIndex),
                             frameWidth, frameHeight, aspectRatio, nonBlack,
                             static_cast<unsigned long long>(hash), center);
            }
            else
            {
                std::fprintf(stderr,
                             "[vulkan-frame] host_frame=%llu readback failed\n",
                             static_cast<unsigned long long>(frameIndex));
            }
            std::fflush(stderr);
        }

        bool present(const std::shared_ptr<GSGpuFrame> &frame,
                     uint32_t frameWidth, uint32_t frameHeight, float aspectRatio)
        {
            static uint64_t frameIndex = 0;
            Util::register_thread_index(0);
            profileFrame(frame, frameWidth, frameHeight, aspectRatio, frameIndex);
            if (!wsi.begin_frame())
                return platform.isAlive();

            auto &device = wsi.get_device();
            auto &destination = device.get_swapchain_view().get_image();
            auto cmd = device.request_command_buffer();
            cmd->swapchain_touch_in_stages(VK_PIPELINE_STAGE_2_TRANSFER_BIT);
            cmd->image_barrier(destination,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_2_NONE, 0,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearValue clear{};
            cmd->clear_image(destination, clear);

            if (frame && frame->GetApi() == GSGpuFrame::Api::Vulkan &&
                frame->GetNativeImage() && frameWidth && frameHeight)
            {
                const auto &source =
                    *static_cast<const Vulkan::Image *>(frame->GetNativeImage());
                const int targetWidth = static_cast<int>(destination.get_width());
                const int targetHeight = static_cast<int>(destination.get_height());
                const float sourceAspect = aspectRatio > 0.0f
                                               ? aspectRatio
                                               : float(frameWidth) / float(frameHeight);
                int displayWidth = targetWidth;
                int displayHeight = std::max(1, int(float(targetWidth) / sourceAspect));
                if (displayHeight > targetHeight)
                {
                    displayHeight = targetHeight;
                    displayWidth = std::max(1, int(float(targetHeight) * sourceAspect));
                }
                const int x = (targetWidth - displayWidth) / 2;
                const int y = (targetHeight - displayHeight) / 2;

                cmd->barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT);
                cmd->blit_image(destination, source,
                                {x, y, 0}, {displayWidth, displayHeight, 1},
                                {0, 0, 0},
                                {static_cast<int>(frameWidth),
                                 static_cast<int>(frameHeight), 1},
                                0, 0);
            }

            cmd->image_barrier(destination,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
            device.submit(cmd);
            ++frameIndex;
            return wsi.end_frame() && platform.isAlive();
        }
    };

    std::unique_ptr<VulkanHost> host;

    bool getGamepadState(int index, XINPUT_STATE &state)
    {
        if (index < 0 || index >= XUSER_MAX_COUNT)
            return false;
        // One pad read asks about twenty buttons, and XInputGetState on an
        // empty slot enumerates devices, which is slow enough to cost frames.
        struct Cached
        {
            XINPUT_STATE state{};
            uint64_t refreshAt = 0;
            bool connected = false;
        };
        thread_local std::array<Cached, XUSER_MAX_COUNT> cache{};
        Cached &cached = cache[static_cast<size_t>(index)];
        const uint64_t now = GetTickCount64();
        if (now >= cached.refreshAt)
        {
            cached.state = {};
            cached.connected = XInputGetState(static_cast<DWORD>(index), &cached.state) == ERROR_SUCCESS;
            cached.refreshAt = now + (cached.connected ? 1 : 1000);
        }
        state = cached.state;
        return cached.connected;
    }

    float normalizeAxis(SHORT value, SHORT deadzone)
    {
        const int magnitude = std::abs(static_cast<int>(value));
        if (magnitude <= deadzone)
            return 0.0f;
        const float normalized =
            float(magnitude - deadzone) / float(32767 - deadzone);
        return std::copysign(std::min(normalized, 1.0f), float(value));
    }
}

bool PS2HostInitializeWindow(const char *title, uint32_t width, uint32_t height)
{
    try
    {
        host = std::make_unique<VulkanHost>(title, width, height);
        return true;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "[host:vulkan] initialization failed: %s\n", e.what());
        host.reset();
        return false;
    }
}

void PS2HostShutdownWindow()
{
    host.reset();
}

bool PS2HostWindowReady()
{
    return host != nullptr;
}

bool PS2HostWindowShouldClose()
{
    return !host || !host->platform.isAlive();
}

void *PS2HostGraphicsDevice()
{
    return host ? &host->wsi.get_device() : nullptr;
}

bool PS2HostPresent(const std::shared_ptr<GSGpuFrame> &frame,
                    uint32_t width, uint32_t height, float aspectRatio)
{
    return host && host->present(frame, width, height, aspectRatio);
}

bool PS2HostKeyDown(PS2HostKey key)
{
    if (!host || !host->platform.hasInputFocus())
        return false;
    const int code = virtualKey(key);
    return (GetAsyncKeyState(code) & 0x8000) != 0 || host->platform.recentlyPressed(code);
}

bool PS2HostKeyPressed(PS2HostKey key)
{
    return host && host->platform.hasInputFocus() &&
           (GetAsyncKeyState(virtualKey(key)) & 1) != 0;
}

bool PS2HostGamepadAvailable(int index)
{
    XINPUT_STATE state{};
    return host && getGamepadState(index, state);
}

bool PS2HostGamepadButtonDown(int index, PS2HostGamepadButton button)
{
    XINPUT_STATE state{};
    if (!host || !getGamepadState(index, state))
        return false;
    if (button == PS2HostGamepadButton::LeftTrigger)
        return state.Gamepad.bLeftTrigger > XINPUT_GAMEPAD_TRIGGER_THRESHOLD;
    if (button == PS2HostGamepadButton::RightTrigger)
        return state.Gamepad.bRightTrigger > XINPUT_GAMEPAD_TRIGGER_THRESHOLD;
    return (state.Gamepad.wButtons & gamepadButton(button)) != 0;
}

float PS2HostGamepadAxisMovement(int index, PS2HostGamepadAxis axis)
{
    XINPUT_STATE state{};
    if (!host || !getGamepadState(index, state))
        return 0.0f;
    switch (axis)
    {
    case PS2HostGamepadAxis::LeftX:
        return normalizeAxis(state.Gamepad.sThumbLX, XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE);
    case PS2HostGamepadAxis::LeftY:
        return -normalizeAxis(state.Gamepad.sThumbLY, XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE);
    case PS2HostGamepadAxis::RightX:
        return normalizeAxis(state.Gamepad.sThumbRX, XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE);
    case PS2HostGamepadAxis::RightY:
        return -normalizeAxis(state.Gamepad.sThumbRY, XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE);
    }
    return 0.0f;
}

#else

namespace
{
    int raylibKey(PS2HostKey key)
    {
        switch (key)
        {
        case PS2HostKey::Up: return KEY_UP;
        case PS2HostKey::Down: return KEY_DOWN;
        case PS2HostKey::Left: return KEY_LEFT;
        case PS2HostKey::Right: return KEY_RIGHT;
        case PS2HostKey::A: return KEY_A;
        case PS2HostKey::C: return KEY_C;
        case PS2HostKey::D: return KEY_D;
        case PS2HostKey::E: return KEY_E;
        case PS2HostKey::Q: return KEY_Q;
        case PS2HostKey::S: return KEY_S;
        case PS2HostKey::V: return KEY_V;
        case PS2HostKey::W: return KEY_W;
        case PS2HostKey::X: return KEY_X;
        case PS2HostKey::Z: return KEY_Z;
        case PS2HostKey::One: return KEY_ONE;
        case PS2HostKey::Three: return KEY_THREE;
        case PS2HostKey::Enter: return KEY_ENTER;
        case PS2HostKey::Escape: return KEY_ESCAPE;
        case PS2HostKey::Space: return KEY_SPACE;
        case PS2HostKey::Tab: return KEY_TAB;
        case PS2HostKey::LeftShift: return KEY_LEFT_SHIFT;
        case PS2HostKey::RightShift: return KEY_RIGHT_SHIFT;
        case PS2HostKey::LeftControl: return KEY_LEFT_CONTROL;
        case PS2HostKey::RightControl: return KEY_RIGHT_CONTROL;
        case PS2HostKey::Keypad0: return KEY_KP_0;
        case PS2HostKey::Keypad1: return KEY_KP_1;
        case PS2HostKey::F1: return KEY_F1;
        }
        return KEY_NULL;
    }

    int raylibButton(PS2HostGamepadButton button)
    {
        switch (button)
        {
        case PS2HostGamepadButton::DpadUp: return GAMEPAD_BUTTON_LEFT_FACE_UP;
        case PS2HostGamepadButton::DpadDown: return GAMEPAD_BUTTON_LEFT_FACE_DOWN;
        case PS2HostGamepadButton::DpadLeft: return GAMEPAD_BUTTON_LEFT_FACE_LEFT;
        case PS2HostGamepadButton::DpadRight: return GAMEPAD_BUTTON_LEFT_FACE_RIGHT;
        case PS2HostGamepadButton::FaceDown: return GAMEPAD_BUTTON_RIGHT_FACE_DOWN;
        case PS2HostGamepadButton::FaceRight: return GAMEPAD_BUTTON_RIGHT_FACE_RIGHT;
        case PS2HostGamepadButton::FaceLeft: return GAMEPAD_BUTTON_RIGHT_FACE_LEFT;
        case PS2HostGamepadButton::FaceUp: return GAMEPAD_BUTTON_RIGHT_FACE_UP;
        case PS2HostGamepadButton::LeftShoulder: return GAMEPAD_BUTTON_LEFT_TRIGGER_1;
        case PS2HostGamepadButton::RightShoulder: return GAMEPAD_BUTTON_RIGHT_TRIGGER_1;
        case PS2HostGamepadButton::LeftTrigger: return GAMEPAD_BUTTON_LEFT_TRIGGER_2;
        case PS2HostGamepadButton::RightTrigger: return GAMEPAD_BUTTON_RIGHT_TRIGGER_2;
        case PS2HostGamepadButton::LeftThumb: return GAMEPAD_BUTTON_LEFT_THUMB;
        case PS2HostGamepadButton::RightThumb: return GAMEPAD_BUTTON_RIGHT_THUMB;
        case PS2HostGamepadButton::Select: return GAMEPAD_BUTTON_MIDDLE_LEFT;
        case PS2HostGamepadButton::Start: return GAMEPAD_BUTTON_MIDDLE_RIGHT;
        }
        return GAMEPAD_BUTTON_UNKNOWN;
    }

    int raylibAxis(PS2HostGamepadAxis axis)
    {
        switch (axis)
        {
        case PS2HostGamepadAxis::LeftX: return GAMEPAD_AXIS_LEFT_X;
        case PS2HostGamepadAxis::LeftY: return GAMEPAD_AXIS_LEFT_Y;
        case PS2HostGamepadAxis::RightX: return GAMEPAD_AXIS_RIGHT_X;
        case PS2HostGamepadAxis::RightY: return GAMEPAD_AXIS_RIGHT_Y;
        }
        return GAMEPAD_AXIS_LEFT_X;
    }
}

bool PS2HostInitializeWindow(const char *title, uint32_t width, uint32_t height)
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(static_cast<int>(width), static_cast<int>(height), title);
    return IsWindowReady();
}

void PS2HostShutdownWindow()
{
    if (IsWindowReady())
        CloseWindow();
}

bool PS2HostWindowReady()
{
    return IsWindowReady();
}

bool PS2HostWindowShouldClose()
{
    return WindowShouldClose();
}

void *PS2HostGraphicsDevice()
{
    return nullptr;
}

bool PS2HostPresent(const std::shared_ptr<GSGpuFrame> &, uint32_t, uint32_t, float)
{
    return false;
}

bool PS2HostKeyDown(PS2HostKey key)
{
    return IsWindowReady() && IsKeyDown(raylibKey(key));
}

bool PS2HostKeyPressed(PS2HostKey key)
{
    return IsWindowReady() && IsKeyPressed(raylibKey(key));
}

bool PS2HostGamepadAvailable(int index)
{
    return IsWindowReady() && IsGamepadAvailable(index);
}

bool PS2HostGamepadButtonDown(int index, PS2HostGamepadButton button)
{
    return IsWindowReady() && IsGamepadButtonDown(index, raylibButton(button));
}

float PS2HostGamepadAxisMovement(int index, PS2HostGamepadAxis axis)
{
    return IsWindowReady() ? GetGamepadAxisMovement(index, raylibAxis(axis)) : 0.0f;
}

#endif
