#include "iop_rpc.h"

#include "../core/iop_cpu.h"
#include "../core/iop_kernel.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

namespace ps2x::iop::detail
{
    namespace
    {
        bool sifCommandTraceEnabled()
        {
            const char *value = std::getenv("PS2X_TRACE_SIF_COMMANDS");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }
    }

    IopRpcBridge::IopRpcBridge(IopHost &host, IopMemory &memory, IopKernel &kernel) noexcept
        : m_host(host), m_memory(memory), m_kernel(kernel)
    {
    }

    void IopRpcBridge::reset()
    {
        m_servers.clear();
        m_commandHandlers.clear();
        m_userCommandBuffer = 0u;
        m_userCommandCount = 0u;
        m_systemCommandBuffer = 0u;
        m_systemCommandCount = 0u;
        std::fill(std::begin(m_sregs), std::end(m_sregs), 0u);
        m_nextDmaId = 1u;
        m_sifInitialized = false;
    }

    bool IopRpcBridge::dispatchSifManImport(uint16_t ordinal,
                                            IopCpuState &cpu,
                                            IopGuestExecutor &executor)
    {
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        switch (ordinal)
        {
        case 4: // sceSifDma2Init
        case 5: // sceSifInit
            m_sifInitialized = true;
            setV0(0u);
            return true;
        case 7:  // sceSifSetDma
        case 32: // sceSifSetDmaIntr
        {
            constexpr uint32_t kDescriptorSize = 16u;
            constexpr uint32_t kMaxDescriptors = 32u;
            const uint32_t descriptorAddress = cpu.gpr[4];
            const uint32_t descriptorCount = cpu.gpr[5];
            if (descriptorAddress == 0u || descriptorCount == 0u || descriptorCount > kMaxDescriptors)
            {
                setV0(0u);
                return true;
            }

            struct PendingTransfer
            {
                uint32_t source = 0u;
                uint32_t destination = 0u;
                uint32_t size = 0u;
            };

            std::array<uint32_t, kMaxDescriptors * 4u> descriptorWords{};
            const size_t descriptorBytes = static_cast<size_t>(descriptorCount) * kDescriptorSize;
            if (!m_memory.readRam(descriptorAddress, descriptorWords.data(), descriptorBytes))
            {
                setV0(0u);
                return true;
            }

            std::array<PendingTransfer, kMaxDescriptors> pending{};
            uint32_t pendingCount = 0u;
            uint32_t largestTransfer = 0u;
            for (uint32_t i = 0u; i < descriptorCount; ++i)
            {
                const uint32_t source = descriptorWords[i * 4u + 0u];
                const uint32_t destination = descriptorWords[i * 4u + 1u];
                const int32_t signedSize = static_cast<int32_t>(descriptorWords[i * 4u + 2u]);
                if (signedSize <= 0)
                    continue;

                const uint32_t size = static_cast<uint32_t>(signedSize);
                if (!m_memory.ownsRamRange(source, size))
                {
                    setV0(0u);
                    return true;
                }
                pending[pendingCount++] = {source, destination, size};
                largestTransfer = std::max(largestTransfer, size);
            }

            // IOP-side sceSifSetDma sends IOP RAM to the EE. Validate all EE
            // destinations before committing any write so a bad chain cannot
            // partially update guest memory, but maybe we could skip this check if we trust the EE-side SIF driver to validate the chain ?!
            // TODO check later
            std::vector<uint8_t> scratch(largestTransfer);
            for (uint32_t i = 0u; i < pendingCount; ++i)
            {
                const PendingTransfer &transfer = pending[i];
                if (!m_host.readGuest(transfer.destination, scratch.data(), transfer.size))
                {
                    setV0(0u);
                    return true;
                }
            }

            for (uint32_t i = 0u; i < pendingCount; ++i)
            {
                const PendingTransfer &transfer = pending[i];
                if (!m_memory.readRam(transfer.source, scratch.data(), transfer.size) || !m_host.writeGuest(transfer.destination, scratch.data(), transfer.size))
                {
                    setV0(0u);
                    return true;
                }
            }

            const uint32_t dmaId = m_nextDmaId++;
            if (m_nextDmaId == 0u || m_nextDmaId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
            {
                m_nextDmaId = 1u;
            }
            if (ordinal == 32u && cpu.gpr[6] != 0u)
            {
                executor.scheduleGuestFunction(cpu.gpr[6],
                                               cpu.gpr[7],
                                               0u,
                                               0u,
                                               0u,
                                               cpu.gpr[28],
                                               1u,
                                               100000u);
            }
            setV0(dmaId);
            return true;
        }
        case 8: // sceSifDmaStat
            setV0(0xFFFFFFFFu);
            return true;
        case 29: // sceSifCheckInit
            setV0(m_sifInitialized ? 1u : 0u);
            return true;
        default:
            setV0(0u);
            return true;
        }
    }

    bool IopRpcBridge::dispatchSifCmdImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        switch (ordinal)
        {
        case 4: // InitCmd
            setV0(0u);
            return true;
        case 5: // ExitCmd
            m_commandHandlers.clear();
            m_userCommandBuffer = 0u;
            m_userCommandCount = 0u;
            m_systemCommandBuffer = 0u;
            m_systemCommandCount = 0u;
            setV0(0u);
            return true;
        case 6: // GetSreg
            setV0(cpu.gpr[4] < std::size(m_sregs) ? m_sregs[cpu.gpr[4]] : 0u);
            return true;
        case 7: // SetSreg
            if (cpu.gpr[4] < std::size(m_sregs))
                m_sregs[cpu.gpr[4]] = cpu.gpr[5];
            setV0(0u);
            return true;
        case 8: // SetCmdBuffer
            m_userCommandBuffer = cpu.gpr[4];
            m_userCommandCount = cpu.gpr[5];
            setV0(0u);
            return true;
        case 9: // SetSysCmdBuffer
            m_systemCommandBuffer = cpu.gpr[4];
            m_systemCommandCount = cpu.gpr[5];
            setV0(0u);
            return true;
        case 10: // AddCmdHandler
        {
            const uint32_t commandId = cpu.gpr[4];
            const uint32_t function = cpu.gpr[5];
            const uint32_t argument = cpu.gpr[6];
            const bool system = (commandId & 0x80000000u) != 0u;
            const uint32_t index = commandId & 0x7FFFFFFFu;
            const uint32_t table = system ? m_systemCommandBuffer : m_userCommandBuffer;
            const uint32_t count = system ? m_systemCommandCount : m_userCommandCount;
            const uint32_t stride = system ? 12u : 8u;

            if (function == 0u)
                m_commandHandlers.erase(commandId);
            else
                m_commandHandlers[commandId] = CommandHandler{function, argument, cpu.gpr[28]};

            if (table != 0u && index < count &&
                m_memory.ownsRamRange(table + index * stride, 8u))
            {
                m_memory.write32(table + index * stride, function);
                m_memory.write32(table + index * stride + 4u, argument);
            }
            if (sifCommandTraceEnabled())
            {
                std::ostringstream message;
                message << "[IOP SIF] "
                        << (function == 0u ? "cleared" : "registered")
                        << " handler cid=0x" << std::hex << commandId
                        << " function=0x" << function
                        << " argument=0x" << argument
                        << " gp=0x" << cpu.gpr[28]
                        << " table=0x" << table
                        << std::dec << " count=" << count;
                m_host.log(LogLevel::Debug, message.str());
            }
            setV0(0u);
            return true;
        }
        case 11: // RemoveCmdHandler
        {
            const uint32_t commandId = cpu.gpr[4];
            const bool system = (commandId & 0x80000000u) != 0u;
            const uint32_t index = commandId & 0x7FFFFFFFu;
            const uint32_t table = system ? m_systemCommandBuffer : m_userCommandBuffer;
            const uint32_t count = system ? m_systemCommandCount : m_userCommandCount;
            const uint32_t stride = system ? 12u : 8u;
            m_commandHandlers.erase(commandId);
            if (table != 0u && index < count &&
                m_memory.ownsRamRange(table + index * stride, 8u))
            {
                m_memory.write32(table + index * stride, 0u);
                m_memory.write32(table + index * stride + 4u, 0u);
            }
            if (sifCommandTraceEnabled())
            {
                std::ostringstream message;
                message << "[IOP SIF] removed handler cid=0x"
                        << std::hex << commandId;
                m_host.log(LogLevel::Debug, message.str());
            }
            setV0(0u);
            return true;
        }
        case 14: // InitRpc
        case 15:
        case 16:
            setV0(0u);
            return true;
        case 12: // sceSifSendCmd
        case 13: // isceSifSendCmd
        {
            constexpr uint32_t kHeaderSize = 16u;
            constexpr uint32_t kMaxPacketSize = 112u;
            const uint32_t commandId = cpu.gpr[4];
            const uint32_t packetAddress = cpu.gpr[5];
            const uint32_t packetSize = cpu.gpr[6];
            const uint32_t extraSource = cpu.gpr[7];
            const uint32_t stackPointer = cpu.gpr[29];
            const uint32_t extraDestination = m_memory.read32(stackPointer + 16u);
            const int32_t signedExtraSize = static_cast<int32_t>(m_memory.read32(stackPointer + 20u));

            if (packetAddress == 0u || packetSize < kHeaderSize || packetSize > kMaxPacketSize ||
                !m_memory.ownsRamRange(packetAddress, packetSize))
            {
                setV0(0u);
                return true;
            }

            std::array<uint8_t, kMaxPacketSize> packet{};
            if (!m_memory.readRam(packetAddress, packet.data(), packetSize))
            {
                setV0(0u);
                return true;
            }

            uint32_t extraSize = 0u;
            if (signedExtraSize > 0)
            {
                extraSize = static_cast<uint32_t>(signedExtraSize);
                if (extraSource == 0u || extraDestination == 0u ||
                    !m_memory.ownsRamRange(extraSource, extraSize) ||
                    !m_host.writeGuest(extraDestination, m_memory.ram().data() + IopMemory::physicalAddress(extraSource), extraSize))
                {
                    setV0(0u);
                    return true;
                }
            }

            const uint32_t sizeWord = packetSize | (extraSize << 8u);
            std::memcpy(packet.data() + 0u, &sizeWord, sizeof(sizeWord));
            std::memcpy(packet.data() + 4u, &extraDestination, sizeof(extraDestination));
            std::memcpy(packet.data() + 8u, &commandId, sizeof(commandId));

            const bool delivered =
                m_host.sendSifCommand(commandId, packet.data(), packetSize);
            if (sifCommandTraceEnabled())
            {
                std::ostringstream message;
                message << "[IOP SIF] sent command to EE cid=0x"
                        << std::hex << commandId
                        << " packet=0x" << packetAddress
                        << " packetSize=0x" << packetSize
                        << " extraSource=0x" << extraSource
                        << " extraDestination=0x" << extraDestination
                        << " extraSize=0x" << extraSize
                        << " delivered=" << (delivered ? 1 : 0);
                m_host.log(LogLevel::Debug, message.str());
            }

            const uint32_t dmaId = m_nextDmaId++;
            if (m_nextDmaId == 0u || m_nextDmaId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
                m_nextDmaId = 1u;
            setV0(dmaId);
            return true;
        }
        case 17: // sceSifRegisterRpc
        {
            RpcServer server;
            server.serverData = cpu.gpr[4];
            server.sid = cpu.gpr[5];
            server.function = cpu.gpr[6];
            server.gp = cpu.gpr[28];
            server.buffer = cpu.gpr[7];
            const uint32_t stackPointer = cpu.gpr[29];
            server.callback = m_memory.read32(stackPointer + 16u);
            server.callbackBuffer = m_memory.read32(stackPointer + 20u);
            server.queue = m_memory.read32(stackPointer + 24u);
            server.priority = m_kernel.currentThreadPriority();
            m_servers[server.sid] = server;
            if (server.serverData != 0u)
            {
                m_memory.write32(server.serverData + 0x20u, server.sid);
                m_memory.write32(server.serverData + 0x28u, server.function);
                m_memory.write32(server.serverData + 0x2Cu, server.buffer);
            }
            setV0(server.serverData);
            return true;
        }
        case 18:
            setV0(0);
            return true;
        case 19: // SetRpcQueue
            setV0(cpu.gpr[4]);
            return true;
        case 20:
        case 21:
            setV0(0);
            return true;
        case 22: // RpcLoop
            m_kernel.sleepCurrent(cpu);
            setV0(0);
            return true;
        case 23:
            setV0(0);
            return true;
        case 24: // RemoveRpc
        {
            const uint32_t serverData = cpu.gpr[4];
            for (auto server = m_servers.begin(); server != m_servers.end(); ++server)
            {
                if (server->second.serverData == serverData)
                {
                    m_servers.erase(server);
                    break;
                }
            }
            setV0(0);
            return true;
        }
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
            setV0(0);
            return true;
        default:
            return false;
        }
    }

    bool IopRpcBridge::handleSifCommand(uint32_t commandId,
                                        const void *packet,
                                        size_t packetSize,
                                        IopGuestExecutor &executor)
    {
        constexpr size_t kHeaderSize = 16u;
        constexpr size_t kMaxPacketSize = 112u;
        if (!packet || packetSize < kHeaderSize || packetSize > kMaxPacketSize)
            return false;

        const auto handler = m_commandHandlers.find(commandId);
        if (handler == m_commandHandlers.end() || handler->second.function == 0u)
            return false;
        const CommandHandler registered = handler->second;

        const uint32_t packetAddress =
            m_memory.allocate(static_cast<uint32_t>(packetSize), 16u);
        if (packetAddress == 0u)
            return false;
        if (!m_memory.writeRam(packetAddress, packet, packetSize))
        {
            static_cast<void>(m_memory.freeAllocation(packetAddress));
            return false;
        }

        const uint32_t result =
            executor.executeGuestFunctionWithBudget(registered.function,
                                                    packetAddress,
                                                    registered.argument,
                                                    0u,
                                                    0u,
                                                    registered.gp,
                                                    100000u);
        const bool freed = m_memory.freeAllocation(packetAddress);
        if (sifCommandTraceEnabled())
        {
            std::ostringstream message;
            message << "[IOP SIF] received command from EE cid=0x"
                    << std::hex << commandId
                    << " handler=0x" << registered.function
                    << " argument=0x" << registered.argument
                    << " gp=0x" << registered.gp
                    << " packetSize=0x" << packetSize
                    << " result=0x" << result
                    << " packetFreed=" << (freed ? 1 : 0);
            m_host.log(LogLevel::Debug, message.str());
        }
        return freed;
    }

    RpcResult IopRpcBridge::handleRpc(const RpcRequest &request, IopGuestExecutor &executor)
    {
        RpcResult result{};
        const auto serverIt = m_servers.find(request.sid);
        if (serverIt == m_servers.end() || serverIt->second.function == 0u)
            return result;

        RpcServer &server = serverIt->second;
        if (request.send.size != 0u && server.buffer != 0u)
        {
            const uint32_t copySize = std::min<uint32_t>(request.send.size, IopMemory::RamSize - std::min(server.buffer, IopMemory::RamSize));
            if (copySize != 0u)
            {
                std::vector<uint8_t> payload(copySize);
                if (m_host.readGuest(request.send.address, payload.data(), payload.size()))
                    (void)m_memory.writeRam(server.buffer, payload.data(), payload.size());
            }
        }

        uint32_t returnPointer = executor.executeGuestFunctionOnThread(
            server.function,
            request.function,
            server.buffer,
            request.send.size,
            0u,
            server.gp,
            server.priority,
            2'000'000u);
        if (returnPointer == 0u)
            returnPointer = server.buffer;
        if (request.receive.address != 0u && request.receive.size != 0u && returnPointer != 0u)
        {
            const uint32_t physical = IopMemory::physicalAddress(returnPointer);
            if (physical < IopMemory::RamSize)
            {
                const uint32_t copySize = std::min<uint32_t>(request.receive.size, IopMemory::RamSize - physical);
                (void)m_host.writeGuest(request.receive.address, m_memory.ram().data() + physical, copySize);
                if (copySize < request.receive.size)
                    (void)m_host.zeroGuest(request.receive.address + copySize, request.receive.size - copySize);
            }
        }

        result.handled = true;
        result.resultAddress = request.receive.address;
        result.serverDispatchPolicy = ServerDispatchPolicy::Suppress;
        result.signalNowaitCompletion = true;
        result.signalCompletion = true;
        return result;
    }

    void IopRpcBridge::onSifTransfer(const SifTransfer &transfer)
    {
        // The EE SIF transport owns the actual directional memory movement.
        // Services still receive both phases through IopSubsystem, but mirroring
        // IOP bytes through an equal-numbered EE address would alias two distinct
        // PS2 address spaces and can overwrite live game data.
        (void)transfer;
    }

    void IopRpcBridge::removeServersInRange(uint32_t base, uint32_t size)
    {
        for (auto server = m_servers.begin(); server != m_servers.end();)
        {
            const uint32_t function = IopMemory::physicalAddress(server->second.function);
            if (function >= base && function < base + size)
                server = m_servers.erase(server);
            else
                ++server;
        }
    }

    bool IopRpcBridge::hasServer(uint32_t sid) const noexcept
    {
        const auto server = m_servers.find(sid);
        return server != m_servers.end() && server->second.function != 0u;
    }
}
