#pragma once

#include "ps2x/iop/iop_types.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace ps2x::iop
{
    class IopHost;
}

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopKernel;
    class IopMemory;

    class IopGuestExecutor
    {
    public:
        virtual ~IopGuestExecutor() = default;

        [[nodiscard]] virtual uint32_t executeGuestFunction(uint32_t address,
                                                            uint32_t a0,
                                                            uint32_t a1,
                                                            uint32_t a2,
                                                            uint32_t a3,
                                                            uint32_t gp) = 0;
        [[nodiscard]] virtual uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                                      uint32_t a0,
                                                                      uint32_t a1,
                                                                      uint32_t a2,
                                                                      uint32_t a3,
                                                                      uint32_t gp,
                                                                      uint32_t instructionBudget)
        {
            return executeGuestFunction(address, a0, a1, a2, a3, gp);
        }
        [[nodiscard]] virtual uint32_t executeGuestFunctionOnThread(
            uint32_t address,
            uint32_t a0,
            uint32_t a1,
            uint32_t a2,
            uint32_t a3,
            uint32_t gp,
            uint32_t priority,
            uint32_t instructionBudget)
        {
            (void)priority;
            return executeGuestFunctionWithBudget(
                address, a0, a1, a2, a3, gp, instructionBudget);
        }
        virtual void scheduleGuestFunction(uint32_t address,
                                           uint32_t a0,
                                           uint32_t a1,
                                           uint32_t a2,
                                           uint32_t a3,
                                           uint32_t gp,
                                           uint64_t delayCycles,
                                           uint32_t instructionBudget)
        {
            (void)delayCycles;
            (void)executeGuestFunctionWithBudget(
                address, a0, a1, a2, a3, gp, instructionBudget);
        }
    };

    class IopRpcBridge
    {
    public:
        IopRpcBridge(IopHost &host, IopMemory &memory, IopKernel &kernel) noexcept;

        void reset();
        [[nodiscard]] bool dispatchSifManImport(uint16_t ordinal,
                                                IopCpuState &cpu,
                                                IopGuestExecutor &executor);
        [[nodiscard]] bool dispatchSifCmdImport(uint16_t ordinal, IopCpuState &cpu);
        [[nodiscard]] bool handleSifCommand(uint32_t commandId,
                                            const void *packet,
                                            size_t packetSize,
                                            IopGuestExecutor &executor);
        [[nodiscard]] RpcResult handleRpc(const RpcRequest &request, IopGuestExecutor &executor);
        void onSifTransfer(const SifTransfer &transfer);
        void removeServersInRange(uint32_t base, uint32_t size);

        [[nodiscard]] bool hasServer(uint32_t sid) const noexcept;
        [[nodiscard]] size_t serverCount() const noexcept { return m_servers.size(); }

    private:
        struct RpcServer
        {
            uint32_t sid = 0;
            uint32_t serverData = 0;
            uint32_t function = 0;
            uint32_t gp = 0;
            uint32_t buffer = 0;
            uint32_t callback = 0;
            uint32_t callbackBuffer = 0;
            uint32_t queue = 0;
            uint32_t priority = 0x40u;
        };

        struct CommandHandler
        {
            uint32_t function = 0;
            uint32_t argument = 0;
            uint32_t gp = 0;
        };

        IopHost &m_host;
        IopMemory &m_memory;
        IopKernel &m_kernel;
        std::unordered_map<uint32_t, RpcServer> m_servers;
        std::unordered_map<uint32_t, CommandHandler> m_commandHandlers;
        uint32_t m_userCommandBuffer = 0;
        uint32_t m_userCommandCount = 0;
        uint32_t m_systemCommandBuffer = 0;
        uint32_t m_systemCommandCount = 0;
        uint32_t m_sregs[32]{};
        uint32_t m_nextDmaId = 1u;
        bool m_sifInitialized = false;
    };
}
