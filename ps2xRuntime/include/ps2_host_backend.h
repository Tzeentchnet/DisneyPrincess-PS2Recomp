#pragma once

#include "raylib.h"
#include <cstdint>
#include <memory>

class GSGpuFrame;

enum class PS2HostKey : uint8_t
{
    Up,
    Down,
    Left,
    Right,
    A,
    C,
    D,
    E,
    Q,
    S,
    V,
    W,
    X,
    Z,
    One,
    Three,
    Enter,
    Escape,
    Space,
    Tab,
    LeftShift,
    RightShift,
    LeftControl,
    RightControl,
    Keypad0,
    Keypad1,
    F1
};

enum class PS2HostGamepadButton : uint8_t
{
    DpadUp,
    DpadDown,
    DpadLeft,
    DpadRight,
    FaceDown,
    FaceRight,
    FaceLeft,
    FaceUp,
    LeftShoulder,
    RightShoulder,
    LeftTrigger,
    RightTrigger,
    LeftThumb,
    RightThumb,
    Select,
    Start
};

enum class PS2HostGamepadAxis : uint8_t
{
    LeftX,
    LeftY,
    RightX,
    RightY
};

bool PS2HostInitializeWindow(const char *title, uint32_t width, uint32_t height);
void PS2HostShutdownWindow();
bool PS2HostWindowReady();
bool PS2HostWindowShouldClose();
void *PS2HostGraphicsDevice();
bool PS2HostPresent(const std::shared_ptr<GSGpuFrame> &frame,
                    uint32_t width, uint32_t height, float aspectRatio);

bool PS2HostKeyDown(PS2HostKey key);
bool PS2HostKeyPressed(PS2HostKey key);
bool PS2HostGamepadAvailable(int index);
bool PS2HostGamepadButtonDown(int index, PS2HostGamepadButton button);
float PS2HostGamepadAxisMovement(int index, PS2HostGamepadAxis axis);
