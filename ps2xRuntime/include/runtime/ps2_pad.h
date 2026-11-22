#ifndef PS2_PAD_H
#define PS2_PAD_H

#include <cstddef>
#include <cstdint>

class PSPadBackend
{
public:
    PSPadBackend() = default;
    ~PSPadBackend() = default;

    static uint16_t mergeActiveLowButtons(uint16_t gamepadButtons, uint16_t keyboardButtons);
    bool readState(int port, int slot, uint8_t *data, size_t size);
};

#endif
