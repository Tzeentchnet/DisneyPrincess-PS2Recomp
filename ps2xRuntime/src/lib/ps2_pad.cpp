#include "runtime/ps2_pad.h"
#include "ps2_host_backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;

    struct ScriptedInputState
    {
        unsigned pressed = 0u;
        unsigned axes[4] = {
            kPadStickCenter,
            kPadStickCenter,
            kPadStickCenter,
            kPadStickCenter,
        };
    };

    // One line: "<pressed mask in hex> [lx ly rx ry]". An unreadable file,
    // for example one being rewritten, keeps the previous input.
    void readScriptedInput(const char *path, ScriptedInputState &state, uint8_t *data)
    {
        if (std::FILE *file = std::fopen(path, "r"))
        {
            unsigned buttons = 0u;
            unsigned next[4] = {kPadStickCenter, kPadStickCenter, kPadStickCenter, kPadStickCenter};
            if (std::fscanf(file, "%x %u %u %u %u", &buttons, &next[0], &next[1], &next[2], &next[3]) >= 1)
            {
                state.pressed = buttons;
                std::copy(next, next + 4, state.axes);
            }
            std::fclose(file);
        }
        data[2] = static_cast<uint8_t>(~state.pressed & 0xFFu);
        data[3] = static_cast<uint8_t>((~state.pressed >> 8) & 0xFFu);
        data[6] = static_cast<uint8_t>(state.axes[0]);
        data[7] = static_cast<uint8_t>(state.axes[1]);
        data[4] = static_cast<uint8_t>(state.axes[2]);
        data[5] = static_cast<uint8_t>(state.axes[3]);
    }
}

uint16_t PSPadBackend::mergeActiveLowButtons(uint16_t gamepadButtons, uint16_t keyboardButtons)
{
    return static_cast<uint16_t>(gamepadButtons & keyboardButtons);
}

bool PSPadBackend::readState(int port, int /*slot*/, uint8_t *data, size_t size)
{
    if (!data || size < 32)
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;

    // Scripted input replaces host input while configured, for repeatable
    // runs that do not depend on window focus.
    const char *const inputFiles[] = {
        std::getenv("PS2X_INPUT_FILE"),
        std::getenv("PS2X_INPUT_FILE_2"),
    };
    const bool scriptedInputEnabled =
        (inputFiles[0] && *inputFiles[0]) ||
        (inputFiles[1] && *inputFiles[1]);
    if (scriptedInputEnabled)
    {
        if (port >= 0 && port < static_cast<int>(std::size(inputFiles)))
        {
            const char *const inputFile = inputFiles[port];
            if (inputFile && *inputFile)
            {
                thread_local ScriptedInputState inputStates[std::size(inputFiles)];
                readScriptedInput(inputFile, inputStates[port], data);
            }
        }
        return true;
    }

    const int kGamepad = port;
    const bool useGamepad = PS2HostGamepadAvailable(kGamepad);
    uint16_t gamepadButtons = 0xFFFFu;
    auto clearGamepadBit = [&gamepadButtons](uint16_t mask)
    { gamepadButtons &= ~mask; };

    if (useGamepad)
    {
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::DpadUp))
            clearGamepadBit(PAD_UP);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::DpadDown))
            clearGamepadBit(PAD_DOWN);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::DpadLeft))
            clearGamepadBit(PAD_LEFT);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::DpadRight))
            clearGamepadBit(PAD_RIGHT);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::FaceDown))
            clearGamepadBit(PAD_CROSS);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::FaceRight))
            clearGamepadBit(PAD_CIRCLE);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::FaceLeft))
            clearGamepadBit(PAD_SQUARE);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::FaceUp))
            clearGamepadBit(PAD_TRIANGLE);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::LeftShoulder))
            clearGamepadBit(PAD_L1);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::RightShoulder))
            clearGamepadBit(PAD_R1);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::LeftTrigger))
            clearGamepadBit(PAD_L2);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::RightTrigger))
            clearGamepadBit(PAD_R2);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::Start))
            clearGamepadBit(PAD_START);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::Select))
            clearGamepadBit(PAD_SELECT);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::LeftThumb))
            clearGamepadBit(PAD_L3);
        if (PS2HostGamepadButtonDown(kGamepad, PS2HostGamepadButton::RightThumb))
            clearGamepadBit(PAD_R3);

        float lx = PS2HostGamepadAxisMovement(kGamepad, PS2HostGamepadAxis::LeftX);
        float ly = PS2HostGamepadAxisMovement(kGamepad, PS2HostGamepadAxis::LeftY);
        float rx = PS2HostGamepadAxisMovement(kGamepad, PS2HostGamepadAxis::RightX);
        float ry = PS2HostGamepadAxisMovement(kGamepad, PS2HostGamepadAxis::RightY);
        data[6] = static_cast<uint8_t>(128 + lx * 127);
        data[7] = static_cast<uint8_t>(128 + ly * 127);
        data[4] = static_cast<uint8_t>(128 + rx * 127);
        data[5] = static_cast<uint8_t>(128 + ry * 127);
    }

    uint16_t keyboardButtons = 0xFFFFu;
    if (port != 0) // The keyboard is the first player's.
    {
        data[2] = static_cast<uint8_t>(gamepadButtons & 0xFF);
        data[3] = static_cast<uint8_t>(gamepadButtons >> 8);
        return true;
    }
    auto clearKeyboardBit = [&keyboardButtons](uint16_t mask)
    { keyboardButtons &= ~mask; };
    if (PS2HostKeyDown(PS2HostKey::Up))
        clearKeyboardBit(PAD_UP);
    if (PS2HostKeyDown(PS2HostKey::Down))
        clearKeyboardBit(PAD_DOWN);
    if (PS2HostKeyDown(PS2HostKey::Left))
        clearKeyboardBit(PAD_LEFT);
    if (PS2HostKeyDown(PS2HostKey::Right))
        clearKeyboardBit(PAD_RIGHT);

    // The arrows are the d-pad and WASD is the left stick.
    const int stickX = (PS2HostKeyDown(PS2HostKey::D) ? 1 : 0) - (PS2HostKeyDown(PS2HostKey::A) ? 1 : 0);
    const int stickY = (PS2HostKeyDown(PS2HostKey::S) ? 1 : 0) - (PS2HostKeyDown(PS2HostKey::W) ? 1 : 0);
    if (stickX != 0 || stickY != 0)
    {
        const int reach = (stickX != 0 && stickY != 0) ? 90 : 127;
        data[6] = static_cast<uint8_t>(128 + stickX * reach);
        data[7] = static_cast<uint8_t>(128 + stickY * reach);
    }
    if (PS2HostKeyDown(PS2HostKey::X) || PS2HostKeyDown(PS2HostKey::Space))
        clearKeyboardBit(PAD_CROSS);
    if (PS2HostKeyDown(PS2HostKey::C) || PS2HostKeyDown(PS2HostKey::Escape))
        clearKeyboardBit(PAD_CIRCLE);
    if (PS2HostKeyDown(PS2HostKey::Z) || PS2HostKeyDown(PS2HostKey::Keypad0))
        clearKeyboardBit(PAD_SQUARE);
    if (PS2HostKeyDown(PS2HostKey::V) || PS2HostKeyDown(PS2HostKey::Keypad1))
        clearKeyboardBit(PAD_TRIANGLE);
    if (PS2HostKeyDown(PS2HostKey::Q))
        clearKeyboardBit(PAD_L1);
    if (PS2HostKeyDown(PS2HostKey::E))
        clearKeyboardBit(PAD_R1);
    if (PS2HostKeyDown(PS2HostKey::LeftShift))
        clearKeyboardBit(PAD_L2);
    if (PS2HostKeyDown(PS2HostKey::RightShift))
        clearKeyboardBit(PAD_R2);
    if (PS2HostKeyDown(PS2HostKey::Enter))
        clearKeyboardBit(PAD_START);
    if (PS2HostKeyDown(PS2HostKey::Tab))
        clearKeyboardBit(PAD_SELECT);

    const uint16_t btns = mergeActiveLowButtons(gamepadButtons, keyboardButtons);
    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    return true;
}
