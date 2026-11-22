#include "iop_stdio.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

namespace ps2x::iop::detail
{
    namespace
    {
        class IopVarArgs
        {
        public:
            IopVarArgs(const IopCpuState &cpu, const IopMemory &memory, uint32_t firstSlot) noexcept
                : m_cpu(cpu), m_memory(memory), m_slot(firstSlot)
            {
            }

            uint32_t next32()
            {
                return readSlot(m_slot++);
            }

            uint64_t next64()
            {
                m_slot = (m_slot + 1u) & ~1u;
                const uint64_t low = readSlot(m_slot++);
                const uint64_t high = readSlot(m_slot++);
                return low | (high << 32u);
            }

        private:
            uint32_t readSlot(uint32_t slot) const
            {
                if (slot < 4u)
                    return m_cpu.gpr[4u + slot];
                return m_memory.read32(m_cpu.gpr[29] + slot * sizeof(uint32_t));
            }

            const IopCpuState &m_cpu;
            const IopMemory &m_memory;
            uint32_t m_slot;
        };

        template <typename T>
        std::string formatValue(const std::string &format, T value)
        {
            std::array<char, 4096> buffer{};
            const int written = std::snprintf(buffer.data(), buffer.size(), format.c_str(), value);
            if (written <= 0)
                return {};
            const size_t length = std::min<size_t>(
                static_cast<size_t>(written), buffer.size() - 1u);
            return std::string(buffer.data(), length);
        }

        std::string renderPrintf(const IopMemory &memory,
                                 const IopCpuState &cpu,
                                 uint32_t formatAddress,
                                 uint32_t firstArgumentSlot)
        {
            const std::string format = memory.readString(formatAddress, 2048u);
            IopVarArgs arguments(cpu, memory, firstArgumentSlot);
            std::string output;
            output.reserve(format.size());

            for (size_t index = 0; index < format.size();)
            {
                if (format[index] != '%')
                {
                    output.push_back(format[index++]);
                    continue;
                }

                const size_t specBegin = index++;
                if (index < format.size() && format[index] == '%')
                {
                    output.push_back('%');
                    ++index;
                    continue;
                }

                std::string flags;
                while (index < format.size() &&
                       std::string_view("-+ #0").find(format[index]) != std::string_view::npos)
                {
                    flags.push_back(format[index++]);
                }

                std::string width;
                if (index < format.size() && format[index] == '*')
                {
                    const int32_t value = static_cast<int32_t>(arguments.next32());
                    if (value < 0)
                    {
                        if (flags.find('-') == std::string::npos)
                            flags.push_back('-');
                        width = std::to_string(-static_cast<int64_t>(value));
                    }
                    else
                    {
                        width = std::to_string(value);
                    }
                    ++index;
                }
                else
                {
                    while (index < format.size() && format[index] >= '0' && format[index] <= '9')
                        width.push_back(format[index++]);
                }

                std::string precision;
                if (index < format.size() && format[index] == '.')
                {
                    precision.push_back(format[index++]);
                    if (index < format.size() && format[index] == '*')
                    {
                        const int32_t value = static_cast<int32_t>(arguments.next32());
                        if (value >= 0)
                            precision += std::to_string(value);
                        else
                            precision.clear();
                        ++index;
                    }
                    else
                    {
                        while (index < format.size() && format[index] >= '0' && format[index] <= '9')
                            precision.push_back(format[index++]);
                    }
                }

                bool wideInteger = false;
                if (index < format.size() && (format[index] == 'h' || format[index] == 'l'))
                {
                    const char length = format[index++];
                    wideInteger = length == 'l' && index < format.size() && format[index] == 'l';
                    if ((length == 'h' || length == 'l') &&
                        index < format.size() && format[index] == length)
                    {
                        ++index;
                    }
                }
                else if (index < format.size() &&
                         std::string_view("jztL").find(format[index]) != std::string_view::npos)
                {
                    wideInteger = format[index] == 'j';
                    ++index;
                }

                if (index >= format.size())
                {
                    output.append(format, specBegin, std::string::npos);
                    break;
                }

                const char conversion = format[index++];
                const std::string baseSpec = "%" + flags + width + precision;
                switch (conversion)
                {
                case 'd':
                case 'i':
                    if (wideInteger)
                    {
                        output += formatValue(baseSpec + "ll" + conversion,
                                              static_cast<long long>(arguments.next64()));
                    }
                    else
                    {
                        output += formatValue(baseSpec + conversion,
                                              static_cast<int32_t>(arguments.next32()));
                    }
                    break;
                case 'u':
                case 'o':
                case 'x':
                case 'X':
                    if (wideInteger)
                    {
                        output += formatValue(baseSpec + "ll" + conversion,
                                              static_cast<unsigned long long>(arguments.next64()));
                    }
                    else
                    {
                        output += formatValue(baseSpec + conversion, arguments.next32());
                    }
                    break;
                case 'c':
                    output += formatValue(baseSpec + conversion,
                                          static_cast<int>(arguments.next32()));
                    break;
                case 's':
                {
                    const uint32_t address = arguments.next32();
                    const std::string value = address == 0u
                                                  ? std::string("(null)")
                                                  : memory.readString(address, 2048u);
                    output += formatValue(baseSpec + conversion, value.c_str());
                    break;
                }
                case 'p':
                {
                    std::string pointerFlags = flags;
                    if (pointerFlags.find('#') == std::string::npos)
                        pointerFlags.push_back('#');
                    output += formatValue("%" + pointerFlags + width + precision + "x",
                                          arguments.next32());
                    break;
                }
                case 'a':
                case 'A':
                case 'e':
                case 'E':
                case 'f':
                case 'F':
                case 'g':
                case 'G':
                {
                    const uint64_t bits = arguments.next64();
                    double value = 0.0;
                    std::memcpy(&value, &bits, sizeof(value));
                    output += formatValue(baseSpec + conversion, value);
                    break;
                }
                default:
                    output.append(format, specBegin, index - specBegin);
                    break;
                }
            }

            return output;
        }
    }

    IopStdio::IopStdio(IopHost &host, IopMemory &memory) noexcept
        : m_host(host), m_memory(memory)
    {
    }

    bool IopStdio::dispatchImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const uint32_t a0 = cpu.gpr[4];
        const uint32_t a1 = cpu.gpr[5];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        const auto logString = [&](std::string_view prefix, uint32_t address, uint32_t resultBias = 0u)
        {
            const std::string text = m_memory.readString(address, 2048u);
            m_host.log(LogLevel::Info, std::string(prefix) + text);
            setV0(static_cast<uint32_t>(text.size()) + resultBias);
        };

        switch (ordinal)
        {
        case 4: // printf
        {
            const std::string text = renderPrintf(m_memory, cpu, a0, 1u);
            m_host.log(LogLevel::Info, std::string("[IOP printf] ") + text);
            setV0(static_cast<uint32_t>(text.size()));
            return true;
        }
        case 5: // getchar
        case 10:
            setV0(0xFFFFFFFFu);
            return true;
        case 6: // putchar
            m_host.log(LogLevel::Info, std::string("[IOP putchar] ") + static_cast<char>(a0 & 0xFFu));
            setV0(a0 & 0xFFu);
            return true;
        case 7: // puts
            logString("[IOP puts] ", a0, 1u);
            return true;
        case 8: // gets
        case 13:
            setV0(0u);
            return true;
        case 9: // fdprintf
        {
            const std::string text = renderPrintf(m_memory, cpu, a1, 2u);
            m_host.log(LogLevel::Info, std::string("[IOP fdprintf] ") + text);
            setV0(static_cast<uint32_t>(text.size()));
            return true;
        }
        case 11:
            setV0(a0 & 0xFFu);
            return true;
        case 12: // fdputs
            logString("[IOP fdputs] ", a0);
            return true;
        case 14: // vfdprintf
            logString("[IOP vfdprintf] ", a1);
            return true;
        default:
            return false;
        }
    }
}
