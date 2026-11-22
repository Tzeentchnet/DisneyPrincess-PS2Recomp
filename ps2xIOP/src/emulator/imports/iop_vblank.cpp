#include "iop_vblank.h"

#include "../core/iop_cpu.h"
#include "../iop_emulator_const.h"
#include "../core/iop_kernel.h"
#include "../services/iop_rpc.h"

#include <algorithm>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr int32_t kKernelErrorFoundHandler = -104;
        constexpr int32_t kKernelErrorNotFoundHandler = -105;
        constexpr uint32_t kHandlerInstructionBudget = 100'000u;

        uint64_t nextEdgeAfter(uint32_t edge, uint64_t currentCycle)
        {
            uint64_t cycle = currentCycle - (currentCycle % kVblankPeriodCycles) +
                             (edge != 0u ? kVblankEndPhaseCycles : 0u);
            while (cycle <= currentCycle)
                cycle += kVblankPeriodCycles;
            return cycle;
        }
    }

    IopVblank::IopVblank(IopKernel &kernel) noexcept
        : m_kernel(kernel)
    {
    }

    bool IopVblank::dispatchImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle)
    {
        switch (ordinal)
        {
        case 4: // WaitVblankStart
        case 5: // WaitVblankEnd
        case 6: // WaitVblank
        case 7: // WaitNonVblank
        {
            const bool waitForEnd = ordinal == 5u || ordinal == 7u;
            const uint64_t phase = waitForEnd ? kVblankEndPhaseCycles : 0u;
            const uint64_t fieldStart = currentCycle - (currentCycle % kVblankPeriodCycles);
            uint64_t wakeCycle = fieldStart + phase;
            if (wakeCycle <= currentCycle)
                wakeCycle += kVblankPeriodCycles;
            m_kernel.delayCurrentUntil(wakeCycle, cpu);
            cpu.gpr[2] = 0u;
            return true;
        }
        case 8: // RegisterVblankHandler(edge, priority, handler, argument)
        {
            const Handler handler{cpu.gpr[4], static_cast<int32_t>(cpu.gpr[5]), cpu.gpr[6], cpu.gpr[7],
                                  cpu.gpr[28], nextEdgeAfter(cpu.gpr[4], currentCycle)};
            const bool exists = std::any_of(m_handlers.begin(), m_handlers.end(),
                                            [&handler](const Handler &other)
                                            { return other.edge == handler.edge && other.function == handler.function; });
            if (exists)
            {
                cpu.gpr[2] = static_cast<uint32_t>(kKernelErrorFoundHandler);
                return true;
            }
            const auto position = std::find_if(m_handlers.begin(), m_handlers.end(),
                                               [&handler](const Handler &other)
                                               { return other.priority > handler.priority; });
            m_handlers.insert(position, handler);
            cpu.gpr[2] = 0u;
            return true;
        }
        case 9: // ReleaseVblankHandler(edge, handler)
        {
            const size_t removed = std::erase_if(m_handlers, [&cpu](const Handler &handler)
                                                 { return handler.edge == cpu.gpr[4] && handler.function == cpu.gpr[5]; });
            cpu.gpr[2] = removed != 0u ? 0u : static_cast<uint32_t>(kKernelErrorNotFoundHandler);
            return true;
        }
        default:
            return false;
        }
    }

    void IopVblank::reset()
    {
        m_handlers.clear();
        m_servicing = false;
    }

    void IopVblank::removeHandlersInRange(uint32_t base, uint32_t size)
    {
        std::erase_if(m_handlers, [base, size](const Handler &handler)
                      { return (handler.function & 0x1FFFFFFFu) - base < size; });
    }

    uint64_t IopVblank::nextEventCycle(uint64_t fallback) const noexcept
    {
        uint64_t next = fallback;
        for (const Handler &handler : m_handlers)
            next = std::min(next, handler.dueCycle);
        return next;
    }

    void IopVblank::serviceDue(uint64_t currentCycle, IopGuestExecutor &executor)
    {
        if (m_servicing || nextEventCycle(UINT64_MAX) > currentCycle)
            return;

        // A handler may register or release handlers, so collect first.
        std::vector<Handler> due;
        for (Handler &handler : m_handlers)
        {
            if (handler.dueCycle > currentCycle)
                continue;
            due.push_back(handler);
            handler.dueCycle = nextEdgeAfter(handler.edge, currentCycle);
        }

        m_servicing = true;
        struct ServiceGuard
        {
            bool &flag;
            ~ServiceGuard() { flag = false; }
        } guard{m_servicing};

        for (const Handler &handler : due)
        {
            const uint32_t keep = executor.executeGuestFunctionWithBudget(
                handler.function, handler.argument, 0u, 0u, 0u, handler.gp, kHandlerInstructionBudget);
            if (keep == 0u)
            {
                std::erase_if(m_handlers, [&handler](const Handler &other)
                              { return other.edge == handler.edge && other.function == handler.function; });
            }
        }
    }
}
