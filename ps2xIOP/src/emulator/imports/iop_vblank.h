#pragma once

#include <cstdint>
#include <vector>

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopGuestExecutor;
    class IopKernel;

    class IopVblank
    {
    public:
        explicit IopVblank(IopKernel &kernel) noexcept;

        void reset();
        [[nodiscard]] bool dispatchImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle);
        void serviceDue(uint64_t currentCycle, IopGuestExecutor &executor);
        [[nodiscard]] uint64_t nextEventCycle(uint64_t fallback) const noexcept;
        void removeHandlersInRange(uint32_t base, uint32_t size);

    private:
        struct Handler
        {
            uint32_t edge = 0; // 0 = vblank start, 1 = vblank end
            int32_t priority = 0;
            uint32_t function = 0;
            uint32_t argument = 0;
            uint32_t gp = 0;
            uint64_t dueCycle = 0;
        };

        IopKernel &m_kernel;
        std::vector<Handler> m_handlers; // in priority order
        bool m_servicing = false;
    };
}
