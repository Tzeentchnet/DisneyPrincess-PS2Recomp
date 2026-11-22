#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace GSMem
{
    class TexturePageCache
    {
    public:
        static constexpr uint32_t kPageSize = 8192u;
        static constexpr uint32_t kEntryCount = 64u;

        TexturePageCache() noexcept
        {
            Invalidate();
        }

        void Invalidate() noexcept
        {
            m_pageBases.fill(UINT32_MAX);
        }

        // byteAddress is the wrapped, swizzled VRAM address. The returned
        // pointer is valid until its cache entry is replaced or invalidated.
        const uint8_t* Resolve(const uint8_t* vram, uint32_t byteAddress) noexcept
        {
            const uint32_t pageBase = byteAddress & ~(kPageSize - 1u);
            const uint32_t entry = (pageBase / kPageSize) & (kEntryCount - 1u);
            if (m_pageBases[entry] != pageBase)
            {
                std::memcpy(m_pages[entry].data(), vram + pageBase, kPageSize);
                m_pageBases[entry] = pageBase;
            }
            return m_pages[entry].data() + (byteAddress & (kPageSize - 1u));
        }

    private:
        static_assert((kEntryCount & (kEntryCount - 1u)) == 0u);
        alignas(64) std::array<std::array<uint8_t, kPageSize>, kEntryCount> m_pages{};
        std::array<uint32_t, kEntryCount> m_pageBases{};
    };
}
