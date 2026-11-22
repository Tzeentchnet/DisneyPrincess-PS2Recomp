#ifndef PS2_ADDRESS_H
#define PS2_ADDRESS_H

#include <cstddef>
#include <cstdint>

inline constexpr uint32_t PS2_RAM_SIZE = 32u * 1024u * 1024u;
inline constexpr uint32_t PS2_RAM_MASK = PS2_RAM_SIZE - 1u;
inline constexpr uint32_t PS2_RAM_BASE = 0x00000000u;
inline constexpr uint32_t PS2_SCRATCHPAD_BASE = 0x70000000u;
inline constexpr uint32_t PS2_SCRATCHPAD_ALIAS_BASE = 0xF0000000u;
inline constexpr uint32_t PS2_SCRATCHPAD_SIZE = 16u * 1024u;
inline constexpr uint32_t PS2_IO_BASE = 0x10000000u;
inline constexpr uint32_t PS2_IO_SIZE = 0x10000u;
inline constexpr uint32_t PS2_BIOS_BASE = 0x1FC00000u;
inline constexpr uint32_t PS2_BIOS_SIZE = 4u * 1024u * 1024u;

inline constexpr uint32_t PS2_VU0_CODE_BASE = 0x11000000u;
inline constexpr uint32_t PS2_VU0_DATA_BASE = 0x11004000u;
inline constexpr uint32_t PS2_VU0_CODE_SIZE = 4u * 1024u;
inline constexpr uint32_t PS2_VU0_DATA_SIZE = 4u * 1024u;

inline constexpr uint32_t PS2_VU1_CODE_BASE = 0x11008000u;
inline constexpr uint32_t PS2_VU1_DATA_BASE = 0x1100C000u;
inline constexpr uint32_t PS2_VU1_MEM_BASE = PS2_VU1_CODE_BASE;
inline constexpr uint32_t PS2_VU1_CODE_SIZE = 16u * 1024u;
inline constexpr uint32_t PS2_VU1_DATA_SIZE = 16u * 1024u;

inline constexpr uint32_t PS2_GS_BASE = 0x12000000u;
inline constexpr uint32_t PS2_GS_PRIV_REG_BASE = PS2_GS_BASE;
inline constexpr uint32_t PS2_GS_PRIV_REG_SIZE = 0x2000u;
inline constexpr size_t PS2_GS_VRAM_SIZE = 4u * 1024u * 1024u;

static_assert((PS2_RAM_SIZE & (PS2_RAM_SIZE - 1u)) == 0u);
static_assert(PS2_RAM_MASK == (PS2_RAM_SIZE - 1u));

static inline constexpr uint32_t PS2_EE_UNCACHED_RAM_MIRROR_BASE = 0x20000000u;
static inline constexpr uint32_t PS2_EE_UNCACHED_RAM_MIRROR_SIZE = 0x20000000u;
static inline constexpr uint32_t PS2_KSEG0_BASE = 0x80000000u;
static inline constexpr uint32_t PS2_KSEG0_KSEG1_SIZE = 0x40000000u;
static inline constexpr uint32_t PS2_KSEG2_BASE = 0xC0000000u;

static inline constexpr bool Ps2AddressInRange(uint32_t value, uint32_t base, uint32_t size)
{
    return (value - base) < size;
}

static inline constexpr bool Ps2IsUncachedRamMirrorAddress(uint32_t addr)
{
    return Ps2AddressInRange(addr, PS2_EE_UNCACHED_RAM_MIRROR_BASE, PS2_EE_UNCACHED_RAM_MIRROR_SIZE);
}

static inline constexpr bool Ps2IsKseg01Address(uint32_t addr)
{
    return Ps2AddressInRange(addr, PS2_KSEG0_BASE, PS2_KSEG0_KSEG1_SIZE);
}

static inline constexpr bool Ps2IsKseg23Address(uint32_t addr)
{
    return addr >= PS2_KSEG2_BASE;
}

static inline constexpr uint32_t Ps2DirectMappedPhysicalAddress(uint32_t addr)
{
    return addr & 0x1FFFFFFFu;
}

static inline constexpr uint32_t Ps2PhysicalAddress(uint32_t addr)
{
    return (addr >= PS2_KSEG0_BASE) ? Ps2DirectMappedPhysicalAddress(addr) : addr;
}

static inline constexpr bool Ps2IsPhysicalSpecialAddress(uint32_t physAddr)
{
    return Ps2AddressInRange(physAddr, PS2_BIOS_BASE, PS2_BIOS_SIZE) ||
           Ps2AddressInRange(physAddr, PS2_SCRATCHPAD_BASE, PS2_SCRATCHPAD_SIZE) ||
           Ps2AddressInRange(physAddr, PS2_IO_BASE, PS2_IO_SIZE) ||
           Ps2AddressInRange(physAddr, PS2_GS_PRIV_REG_BASE, PS2_GS_PRIV_REG_SIZE) ||
           (physAddr >= PS2_VU0_DATA_BASE && physAddr < (PS2_VU1_CODE_BASE + PS2_VU1_CODE_SIZE));
}

static inline constexpr bool Ps2IsSpecialAddress(uint32_t addr)
{
    if (Ps2IsKseg23Address(addr))
        return true;

    return Ps2IsPhysicalSpecialAddress(Ps2PhysicalAddress(addr));
}

#endif // PS2_ADDRESS_H
