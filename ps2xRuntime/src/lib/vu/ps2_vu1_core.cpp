#include "runtime/ps2_vu1.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <ps2_log.h>

#if defined(_MSC_VER)
#include <intrin.h>
#define VU_FORCE_INLINE __forceinline
#define VU_LAMBDA_INLINE
#elif defined(USE_SSE2NEON)
#include "sse2neon.h"
#define VU_FORCE_INLINE inline __attribute__((always_inline))
#define VU_LAMBDA_INLINE __attribute__((always_inline))
#else
#include <smmintrin.h>
#define VU_FORCE_INLINE inline __attribute__((always_inline))
#define VU_LAMBDA_INLINE __attribute__((always_inline))
#endif

namespace
{
    // movemask numbers x as bit 0; DEST and the MAC nibbles number it as bit 3.
    constexpr uint8_t kReverseLanes[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

    alignas(16) constexpr uint32_t kLaneSelect[16][4] = {
#define VU_LANES(m) {((m) & 1) ? ~0u : 0u, ((m) & 2) ? ~0u : 0u, ((m) & 4) ? ~0u : 0u, ((m) & 8) ? ~0u : 0u}
        VU_LANES(0), VU_LANES(1), VU_LANES(2), VU_LANES(3),
        VU_LANES(4), VU_LANES(5), VU_LANES(6), VU_LANES(7),
        VU_LANES(8), VU_LANES(9), VU_LANES(10), VU_LANES(11),
        VU_LANES(12), VU_LANES(13), VU_LANES(14), VU_LANES(15),
#undef VU_LANES
    };

    VU_FORCE_INLINE int laneBits(__m128i mask)
    {
        return _mm_movemask_ps(_mm_castsi128_ps(mask));
    }

    // normalizeOperand on four lanes.
    VU_FORCE_INLINE __m128 normalizeLanes(__m128 value)
    {
        const __m128i bits = _mm_castps_si128(value);
        const __m128i exponentMask = _mm_set1_epi32(0x7F800000);
        const __m128i exponent = _mm_and_si128(bits, exponentMask);
        const __m128i sign = _mm_and_si128(bits, _mm_set1_epi32(static_cast<int>(0x80000000u)));
        __m128i out = _mm_blendv_epi8(bits, sign, _mm_cmpeq_epi32(exponent, _mm_setzero_si128()));
        out = _mm_blendv_epi8(out, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF)),
                              _mm_cmpeq_epi32(exponent, exponentMask));
        return _mm_castsi128_ps(out);
    }

    // Lanes holding a normal value strictly between the given magnitudes.
    VU_FORCE_INLINE int lanesInRange(__m128i magnitude, int32_t above, int32_t below)
    {
        return laneBits(_mm_and_si128(_mm_cmpgt_epi32(magnitude, _mm_set1_epi32(above)),
                                      _mm_cmpgt_epi32(_mm_set1_epi32(below), magnitude)));
    }

    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    constexpr uint32_t kTraceDisabled = std::numeric_limits<uint32_t>::max();

    uint32_t readTraceTarget(const char *environmentName)
    {
        const char *const value = std::getenv(environmentName);
        if (!value || *value == '\0')
            return kTraceDisabled;
        char *end = nullptr;
        const unsigned long target = std::strtoul(value, &end, 0);
        return end != value && *end == '\0' ? static_cast<uint32_t>(target) : kTraceDisabled;
    }

    const uint32_t g_traceStoreQword = readTraceTarget("PS2X_TRACE_VU_STORE_QWORD");
    const uint32_t g_traceVfRegister = readTraceTarget("PS2X_TRACE_VU_VF");
    const uint32_t g_traceViRegister = readTraceTarget("PS2X_TRACE_VU_VI");

    // Claims the lowest free slot of an N-entry pipeline; -1 when full.
    template <uint32_t N>
    int claimSlot(uint32_t &mask)
    {
        const uint32_t freeSlots = ~mask & ((1u << N) - 1u);
        if (freeSlots == 0u)
            return -1;
        const int slot = std::countr_zero(freeSlots);
        mask |= 1u << slot;
        return slot;
    }

    // What an FMAC upper instruction computes per lane, from its normalized
    // operands: A is VS (VS crossed for OPMULA/OPMSUB), B the broadcast lane,
    // Q, I, VT, or VT crossed. Mirrors calculateFmacExactResult.
    enum class FmacOp : uint8_t
    {
        None,
        Add,      // A + B
        Sub,      // A - B
        Mul,      // A * B
        MAdd,     // ACC + A * B
        MSub,     // ACC - A * B
        OuterMul, // A * B, w = 0 (OPMULA)
        OuterMSub // ACC - A * B, w = 0 (OPMSUB)
    };

    enum class FmacSource : uint8_t
    {
        Broadcast,
        Q,
        I,
        Full,
        Cross,
    };

    struct FmacDecode
    {
        FmacOp op = FmacOp::None;
        FmacSource source = FmacSource::Full;
        uint8_t broadcast = 0;
    };

    FmacDecode decodeFmac(uint32_t upper)
    {
        const uint32_t op = upper & 0x3Fu;
        const bool special = op >= 0x3Cu;
        const uint32_t code = special ? ((upper & 3u) | ((upper >> 4) & 0x7Cu)) : op;
        const uint8_t bc = static_cast<uint8_t>(code & 3u);
        if (code <= 0x03u)
            return {FmacOp::Add, FmacSource::Broadcast, bc};
        if (code <= 0x07u)
            return {FmacOp::Sub, FmacSource::Broadcast, bc};
        if (code <= 0x0Bu)
            return {FmacOp::MAdd, FmacSource::Broadcast, bc};
        if (code <= 0x0Fu)
            return {FmacOp::MSub, FmacSource::Broadcast, bc};
        if (code >= 0x18u && code <= 0x1Bu)
            return {FmacOp::Mul, FmacSource::Broadcast, bc};
        switch (code)
        {
        case 0x1Cu: return {FmacOp::Mul, FmacSource::Q};
        case 0x1Eu: return {FmacOp::Mul, FmacSource::I};
        case 0x20u: return {FmacOp::Add, FmacSource::Q};
        case 0x21u: return {FmacOp::MAdd, FmacSource::Q};
        case 0x22u: return {FmacOp::Add, FmacSource::I};
        case 0x23u: return {FmacOp::MAdd, FmacSource::I};
        case 0x24u: return {FmacOp::Sub, FmacSource::Q};
        case 0x25u: return {FmacOp::MSub, FmacSource::Q};
        case 0x26u: return {FmacOp::Sub, FmacSource::I};
        case 0x27u: return {FmacOp::MSub, FmacSource::I};
        case 0x28u: return {FmacOp::Add, FmacSource::Full};
        case 0x29u: return {FmacOp::MAdd, FmacSource::Full};
        case 0x2Au: return {FmacOp::Mul, FmacSource::Full};
        case 0x2Cu: return {FmacOp::Sub, FmacSource::Full};
        case 0x2Du: return {FmacOp::MSub, FmacSource::Full};
        case 0x2Eu: return {special ? FmacOp::OuterMul : FmacOp::OuterMSub, FmacSource::Cross};
        default: return {};
        }
    }

    constexpr bool isProductSum(FmacOp op)
    {
        return op == FmacOp::MAdd || op == FmacOp::MSub || op == FmacOp::OuterMSub;
    }

    // Decoded upper operation. FMAC kinds are operation * 5 + source.
    namespace UpperKind
    {
        constexpr uint8_t Generic = 0;
        constexpr uint8_t Nop = 1;
        constexpr uint8_t MaxBroadcast = 2;
        constexpr uint8_t MaxI = 3;
        constexpr uint8_t Max = 4;
        constexpr uint8_t MiniBroadcast = 5;
        constexpr uint8_t MiniI = 6;
        constexpr uint8_t Mini = 7;
        constexpr uint8_t Abs = 8;
        constexpr uint8_t IntToFloat = 9;
        constexpr uint8_t FloatToInt = 10;
        constexpr uint8_t Clip = 11;
        constexpr uint8_t Fmac = 16;
        constexpr uint8_t WritesAcc = 64;

        constexpr uint8_t fmac(FmacOp op, FmacSource source, bool acc)
        {
            return static_cast<uint8_t>(Fmac + (static_cast<uint8_t>(op) - 1u) * 5u + static_cast<uint8_t>(source) +
                                        (acc ? WritesAcc : 0u));
        }
    }

    namespace LowerKind
    {
        constexpr uint8_t Generic = 0;
        constexpr uint8_t Nop = 1;
        constexpr uint8_t Lq = 2;
        constexpr uint8_t Sq = 3;
        constexpr uint8_t Lqi = 4;
        constexpr uint8_t Sqi = 5;
        constexpr uint8_t Ilw = 6;
        constexpr uint8_t Isw = 7;
        constexpr uint8_t AddImmediate = 8;
        constexpr uint8_t Iadd = 9;
        constexpr uint8_t Iand = 10;
        constexpr uint8_t Ior = 11;
        constexpr uint8_t Move = 12;
        constexpr uint8_t Mr32 = 13;
        constexpr uint8_t Mtir = 14;
        constexpr uint8_t Branch = 15;
        constexpr uint8_t BranchEqual = 16;
        constexpr uint8_t BranchNotEqual = 17;
        constexpr uint8_t ClipAnd = 18;
    }

    // Scale of ITOF/FTOI 0, 4, 12 and 15.
    constexpr float kFixedScale[4] = {1.0f, 16.0f, 4096.0f, 32768.0f};
    constexpr float kFixedInverse[4] = {1.0f, 1.0f / 16.0f, 1.0f / 4096.0f, 1.0f / 32768.0f};

    alignas(16) constexpr uint8_t kBroadcastLane[4][16] = {
        {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
        {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7},
        {8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11},
        {12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15},
    };

    VU_FORCE_INLINE __m128 broadcastLane(__m128 value, uint32_t lane)
    {
        return _mm_castsi128_ps(_mm_shuffle_epi8(
            _mm_castps_si128(value), _mm_load_si128(reinterpret_cast<const __m128i *>(kBroadcastLane[lane]))));
    }

    VU_FORCE_INLINE void writeLanes(float *destination, __m128 value, uint32_t lanes)
    {
        const __m128 select = _mm_load_ps(reinterpret_cast<const float *>(kLaneSelect[lanes]));
        _mm_storeu_ps(destination, _mm_blendv_ps(_mm_loadu_ps(destination), value, select));
    }

    inline int nextSlot(uint32_t &bits)
    {
        const int slot = std::countr_zero(bits);
        bits &= bits - 1u;
        return slot;
    }
}

void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    reset();
}

VU_FORCE_INLINE VU1Interpreter::FlagPipelineEntry *VU1Interpreter::pushFlags()
{
    if (m_flagCount == kMaxFlagEntries) [[unlikely]]
    {
        commitFlags();
        if (m_flagCount == kMaxFlagEntries)
            return nullptr;
    }
    FlagPipelineEntry &entry = m_flagPipeline[(m_flagHead + m_flagCount++) % kMaxFlagEntries];
    entry = {};
    entry.issueCycle = m_cycle;
    entry.readyCycle = m_cycle + kFmacLatency;
    return &entry;
}

VU_FORCE_INLINE void VU1Interpreter::commitFlags()
{
    while (m_flagCount != 0u)
    {
        const FlagPipelineEntry &entry = m_flagPipeline[m_flagHead];
        if (entry.readyCycle > m_cycle)
            return;
        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        if (entry.writesClip)
            m_state.clip = entry.clip;
        m_flagHead = (m_flagHead + 1u) % kMaxFlagEntries;
        --m_flagCount;
    }
}

// The common FMAC case: every destination lane's float result is already the
// exact one after clamping, so its flags follow from sign and zero alone.
// Anything near a range limit, or a cancellation to zero, takes applyFmacDest.
template <uint8_t Kind>
VU_FORCE_INLINE void VU1Interpreter::execFmac(const UpperInstruction &upper, uint32_t instruction)
{
    constexpr uint8_t index = (Kind & ~UpperKind::WritesAcc) - UpperKind::Fmac;
    constexpr bool writesAcc = (Kind & UpperKind::WritesAcc) != 0u;
    constexpr FmacOp op = static_cast<FmacOp>(index / 5u + 1u);
    constexpr FmacSource source = static_cast<FmacSource>(index % 5u);
    constexpr bool outer = source == FmacSource::Cross;

    const __m128 vs = normalizeLanes(_mm_loadu_ps(m_state.vf[upper.fs]));
    const __m128 vt = normalizeLanes(_mm_loadu_ps(m_state.vf[upper.ft]));
    __m128 a = vs;
    __m128 b = vt;
    float scalar = 0.0f;
    if constexpr (source == FmacSource::Broadcast)
    {
        b = broadcastLane(vt, upper.broadcast);
    }
    else if constexpr (source == FmacSource::Q || source == FmacSource::I)
    {
        scalar = normalizeOperand(source == FmacSource::Q ? m_state.q : m_state.i);
        b = _mm_set1_ps(scalar);
    }
    else if constexpr (outer)
    {
        a = _mm_shuffle_ps(vs, vs, _MM_SHUFFLE(3, 0, 2, 1));
        b = _mm_shuffle_ps(vt, vt, _MM_SHUFFLE(3, 1, 0, 2));
    }

    constexpr int32_t kBelowNormal = 0x007FFFFF;
    constexpr int32_t kFloatMax = 0x7F7FFFFF;
    const __m128i signMask = _mm_set1_epi32(static_cast<int>(0x80000000u));
    const __m128i none = _mm_setzero_si128();

    __m128 acc = _mm_setzero_ps();
    __m128 result;
    int exactLanes = 0xF;
    int exactZero = 0;
    uint32_t productSticky = 0u;
    constexpr bool adds = op == FmacOp::Add || op == FmacOp::Sub;
    constexpr bool sum = isProductSum(op);

    if (op == FmacOp::Add)
    {
        result = _mm_add_ps(a, b);
    }
    else if (op == FmacOp::Sub)
    {
        result = _mm_sub_ps(a, b);
    }
    else
    {
        const __m128 product = _mm_mul_ps(a, b);
        const __m128i productMagnitude = _mm_andnot_si128(signMask, _mm_castps_si128(product));
        // A zero product is exact only when an operand is zero.
        exactZero = _mm_movemask_ps(_mm_or_ps(_mm_cmpeq_ps(a, _mm_setzero_ps()),
                                              _mm_cmpeq_ps(b, _mm_setzero_ps())));
        if (sum)
        {
            exactLanes = lanesInRange(productMagnitude, kBelowNormal, kFloatMax) | exactZero;
            if ((exactZero & upper.lanes) != 0)
                productSticky |= 0x1u;
            if ((_mm_movemask_ps(product) & upper.lanes) != 0)
                productSticky |= 0x2u;

            acc = normalizeLanes(_mm_loadu_ps(m_state.acc));
            result = op == FmacOp::MAdd ? _mm_add_ps(acc, product) : _mm_sub_ps(acc, product);
        }
        else
        {
            result = product;
        }
        if (outer)
        {
            result = _mm_and_ps(result, _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0)));
        }
    }

    const __m128i magnitude = _mm_andnot_si128(signMask, _mm_castps_si128(result));
    const int zero = laneBits(_mm_cmpeq_epi32(magnitude, none));
    int safe;
    if (adds)
    {
        safe = lanesInRange(magnitude, kBelowNormal, kFloatMax) | zero;
    }
    else
    {
        // A product sum rounds twice. Between 2^-100 and 2^100 that can move
        // neither its sign nor its range.
        safe = sum ? lanesInRange(magnitude, 0x0D800000, 0x71800000)
                   : lanesInRange(magnitude, kBelowNormal, kFloatMax);
        safe |= zero & exactZero;
        if (outer)
            safe |= 0x8;
        safe &= exactLanes;
    }

    float *const destination = writesAcc ? m_state.acc : m_state.vf[upper.destination];
    if ((safe & upper.lanes) != upper.lanes) [[unlikely]]
    {
        float vsLanes[4], vtLanes[4], accLanes[4], resultLanes[4];
        _mm_storeu_ps(vsLanes, vs);
        _mm_storeu_ps(vtLanes, vt);
        _mm_storeu_ps(accLanes, acc);
        _mm_storeu_ps(resultLanes, result);
        m_currentUpperInstruction = instruction;
        m_fmacVs = vsLanes;
        m_fmacVt = vtLanes;
        m_fmacAcc = accLanes;
        m_fmacQ = m_fmacI = scalar;
        if (writesAcc)
            applyFmacDestAcc(resultLanes, DEST(instruction));
        else
            applyFmacDest(destination, resultLanes, DEST(instruction));
        return;
    }
    if (upper.lanes == 0u)
        return;

    FlagPipelineEntry *const flags = pushFlags();
    if (!flags) [[unlikely]]
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
    }
    else
    {
        const int zeroLanes = zero & upper.lanes;
        const int negativeLanes = _mm_movemask_ps(result) & upper.lanes;
        flags->mac = kReverseLanes[zeroLanes] | (static_cast<uint32_t>(kReverseLanes[negativeLanes]) << 4);
        flags->status = (zeroLanes != 0 ? 0x1u : 0u) | (negativeLanes != 0 ? 0x2u : 0u);
        flags->extraSticky = productSticky;
        flags->writesMac = true;
        flags->writesStatus = true;
    }

    writeLanes(destination, result, upper.lanes);
}

VU_FORCE_INLINE void VU1Interpreter::execUpperDecoded(const DecodedInstructionPair &decoded)
{
    const UpperInstruction &op = decoded.upperOp;
    float *const destination = m_state.vf[op.destination];
    const auto vs = [&]() VU_LAMBDA_INLINE { return normalizeLanes(_mm_loadu_ps(m_state.vf[op.fs])); };
    const auto vt = [&]() VU_LAMBDA_INLINE { return normalizeLanes(_mm_loadu_ps(m_state.vf[op.ft])); };

    switch (op.kind)
    {
    case UpperKind::Nop:
        return;
#define VU_FMAC(operation, source)                                                      \
    case UpperKind::fmac(FmacOp::operation, FmacSource::source, false):                 \
        execFmac<UpperKind::fmac(FmacOp::operation, FmacSource::source, false)>(op, decoded.upper); \
        return;                                                                         \
    case UpperKind::fmac(FmacOp::operation, FmacSource::source, true):                  \
        execFmac<UpperKind::fmac(FmacOp::operation, FmacSource::source, true)>(op, decoded.upper); \
        return;
#define VU_FMAC_SOURCES(operation) \
    VU_FMAC(operation, Broadcast)  \
    VU_FMAC(operation, Q)          \
    VU_FMAC(operation, I)          \
    VU_FMAC(operation, Full)
        VU_FMAC_SOURCES(Add)
        VU_FMAC_SOURCES(Sub)
        VU_FMAC_SOURCES(Mul)
        VU_FMAC_SOURCES(MAdd)
        VU_FMAC_SOURCES(MSub)
        VU_FMAC(OuterMul, Cross)
        VU_FMAC(OuterMSub, Cross)
#undef VU_FMAC_SOURCES
#undef VU_FMAC
    // MAXPS and MINPS return the second operand unless the comparison holds,
    // which is exactly (a > b ? a : b) and (a < b ? a : b).
    case UpperKind::MaxBroadcast:
        writeLanes(destination, _mm_max_ps(vs(), broadcastLane(vt(), op.broadcast)), op.lanes);
        return;
    case UpperKind::MaxI:
        writeLanes(destination, _mm_max_ps(vs(), _mm_set1_ps(normalizeOperand(m_state.i))), op.lanes);
        return;
    case UpperKind::Max:
        writeLanes(destination, _mm_max_ps(vs(), vt()), op.lanes);
        return;
    case UpperKind::MiniBroadcast:
        writeLanes(destination, _mm_min_ps(vs(), broadcastLane(vt(), op.broadcast)), op.lanes);
        return;
    case UpperKind::MiniI:
        writeLanes(destination, _mm_min_ps(vs(), _mm_set1_ps(normalizeOperand(m_state.i))), op.lanes);
        return;
    case UpperKind::Mini:
        writeLanes(destination, _mm_min_ps(vs(), vt()), op.lanes);
        return;
    case UpperKind::Abs:
        writeLanes(destination, _mm_andnot_ps(_mm_set1_ps(-0.0f), vs()), op.lanes);
        return;
    case UpperKind::IntToFloat:
        writeLanes(destination,
                   _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(m_state.vf[op.fs]))),
                              _mm_set1_ps(kFixedInverse[op.broadcast])),
                   op.lanes);
        return;
    case UpperKind::FloatToInt:
    {
        const __m128 scaled = _mm_mul_ps(vs(), _mm_set1_ps(kFixedScale[op.broadcast]));
        const __m128i tooLarge = _mm_cmpgt_epi32(_mm_castps_si128(scaled), _mm_set1_epi32(0x4EFFFFFF));
        writeLanes(destination, _mm_castsi128_ps(_mm_xor_si128(_mm_cvttps_epi32(scaled), tooLarge)), op.lanes);
        return;
    }
    case UpperKind::Clip:
    {
        uint32_t w = 0u;
        std::memcpy(&w, &m_state.vf[op.ft][3], sizeof(w));
        const __m128i limit = _mm_set1_epi32(
            (w & 0x7F800000u) != 0u ? static_cast<int32_t>(w & 0x7FFFFFFFu) : 0x007FFFFF);
        const __m128i value = _mm_loadu_si128(reinterpret_cast<const __m128i *>(m_state.vf[op.fs]));
        const int positive = laneBits(_mm_cmpgt_epi32(value, limit));
        const int negative = laneBits(_mm_cmpgt_epi32(
            _mm_xor_si128(value, _mm_set1_epi32(static_cast<int>(0x80000000u))), limit));
        queueClip(static_cast<uint32_t>((positive & 1) | ((negative & 1) << 1) |
                                        ((positive & 2) << 1) | ((negative & 2) << 2) |
                                        ((positive & 4) << 2) | ((negative & 4) << 3)));
        return;
    }
    default:
        execUpper(decoded.upper);
        return;
    }
}

VU_FORCE_INLINE void VU1Interpreter::execLowerDecoded(const DecodedInstructionPair &decoded, uint8_t *vuData,
                                                      uint32_t dataSize, GS &gs, PS2Memory *memory)
{
    const LowerInstruction &op = decoded.lowerOp;
    // Data memory is 4 or 16 KiB, so a masked qword address is always in range.
    const uint32_t addressMask = dataSize - 1u;
    const auto relativeAddress = [&]() VU_LAMBDA_INLINE
    {
        return (static_cast<uint32_t>(m_state.vi[op.is] + op.immediate) * 16u) & addressMask;
    };
    const auto pointerAddress = [&]() VU_LAMBDA_INLINE
    {
        return (static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[op.is])) * 16u) & addressMask;
    };
    const auto takeBranch = [&]() VU_LAMBDA_INLINE
    {
        m_state.branchPending = true;
        m_state.branchTarget = static_cast<uint32_t>(m_state.pc + 8 + op.immediate * 8) & microAddressMask();
        m_state.branchDelay = 1;
    };

    switch (op.kind)
    {
    case LowerKind::Nop:
        return;
    case LowerKind::Lq:
        writeLanes(m_state.vf[op.it],
                   _mm_loadu_ps(reinterpret_cast<const float *>(vuData + relativeAddress())), op.lanes);
        return;
    case LowerKind::Sq:
        writeLanes(reinterpret_cast<float *>(vuData + relativeAddress()), _mm_loadu_ps(m_state.vf[op.it]), op.lanes);
        return;
    case LowerKind::Lqi:
        writeLanes(m_state.vf[op.it],
                   _mm_loadu_ps(reinterpret_cast<const float *>(vuData + pointerAddress())), op.lanes);
        if (op.is != 0u)
            m_state.vi[op.is] = static_cast<int16_t>(m_state.vi[op.is] + 1);
        return;
    case LowerKind::Sqi:
        writeLanes(reinterpret_cast<float *>(vuData + pointerAddress()), _mm_loadu_ps(m_state.vf[op.it]), op.lanes);
        if (op.is != 0u)
            m_state.vi[op.is] = static_cast<int16_t>(m_state.vi[op.is] + 1);
        return;
    case LowerKind::Ilw:
    {
        uint32_t value = 0u;
        std::memcpy(&value, vuData + relativeAddress() + op.lanes * 4u, sizeof(value));
        m_state.vi[op.it] = static_cast<int16_t>(value);
        return;
    }
    case LowerKind::Isw:
        writeLanes(reinterpret_cast<float *>(vuData + relativeAddress()),
                   _mm_castsi128_ps(_mm_set1_epi32(static_cast<uint16_t>(m_state.vi[op.it]))), op.lanes);
        return;
    case LowerKind::AddImmediate:
        m_state.vi[op.it] = static_cast<int16_t>(m_state.vi[op.is] + op.immediate);
        return;
    case LowerKind::Iadd:
        m_state.vi[op.id] = static_cast<int16_t>(m_state.vi[op.is] + m_state.vi[op.it]);
        return;
    case LowerKind::Iand:
        m_state.vi[op.id] = m_state.vi[op.is] & m_state.vi[op.it];
        return;
    case LowerKind::Ior:
        m_state.vi[op.id] = m_state.vi[op.is] | m_state.vi[op.it];
        return;
    case LowerKind::Move:
        writeLanes(m_state.vf[op.it], _mm_loadu_ps(m_state.vf[op.is]), op.lanes);
        return;
    case LowerKind::Mr32:
    {
        const __m128 source = _mm_loadu_ps(m_state.vf[op.is]);
        writeLanes(m_state.vf[op.it], _mm_shuffle_ps(source, source, _MM_SHUFFLE(0, 3, 2, 1)), op.lanes);
        return;
    }
    case LowerKind::Mtir:
    {
        uint32_t value = 0u;
        std::memcpy(&value, &m_state.vf[op.is][op.lanes], sizeof(value));
        m_state.vi[op.it] = static_cast<int16_t>(value);
        return;
    }
    case LowerKind::Branch:
        takeBranch();
        return;
    case LowerKind::BranchEqual:
        if (static_cast<int16_t>(readBranchVi(op.is)) == static_cast<int16_t>(readBranchVi(op.it)))
            takeBranch();
        return;
    case LowerKind::BranchNotEqual:
        if (static_cast<int16_t>(readBranchVi(op.is)) != static_cast<int16_t>(readBranchVi(op.it)))
            takeBranch();
        return;
    case LowerKind::ClipAnd:
        commitFlags();
        m_state.vi[1] = (m_state.clip & decoded.lower & 0xFFFFFFu) != 0u ? 1 : 0;
        return;
    default:
        if (decoded.readsFlags)
            commitFlags();
        execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        return;
    }
}


VU1XgkickDebugState VU1Interpreter::xgkickDebugState() const
{
    return {
        m_xgkick.active,
        m_xgkick.sourceAddress,
        m_xgkick.totalBytes,
        m_xgkick.copiedBytes,
        m_xgkick.currentTagEnd,
        m_xgkick.cycleCredit,
        m_xgkick.issueCycle,
        m_xgkick.currentTagEop
    };
}

void VU1Interpreter::resetScheduler()
{
    m_flagHead = 0;
    m_flagCount = 0;
    m_fdiv = {};
    m_efu = {};
    // The packet buffer is only read up to copiedBytes; clearing its 64 KiB
    // for each program and kick would cost more than most kicks.
    m_xgkick.active = false;
    m_xgkick.sourceAddress = 0;
    m_xgkick.totalBytes = 0;
    m_xgkick.copiedBytes = 0;
    m_xgkick.currentTagEnd = 0;
    m_xgkick.cycleCredit = 0;
    m_xgkick.issueCycle = 0;
    m_xgkick.currentTagEop = false;
    m_issued = {};
    m_efuResourceReady = 0;
    m_nextCommitCycle = std::numeric_limits<uint64_t>::max();
    m_writebackHorizon = 0;
    m_efuMask = 0;
    m_workingClip = m_state.clip;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    resetScheduler();
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4], uint32_t &productSticky)
{
    productSticky = 0u;
    const FmacDecode fmac = decodeFmac(m_currentUpperInstruction);
    const bool single = fmac.op == FmacOp::Add || fmac.op == FmacOp::Sub ||
                        fmac.op == FmacOp::Mul || fmac.op == FmacOp::OuterMul;
    const bool productSum = isProductSum(fmac.op);

    // Operands as calculateFmacExactResult derives them, from the values
    // execUpper already normalized.
    bool haveOperands = false;
    double a[4]{}, b[4]{}, acc[4]{};
    auto gatherOperands = [&]()
    {
        if (haveOperands)
            return;
        haveOperands = true;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const bool cross = fmac.source == FmacSource::Cross;
        for (uint32_t lane = 0; lane < 4u; ++lane)
        {
            a[lane] = m_fmacVs[cross ? crossLeft[lane] : lane];
            if (productSum)
                acc[lane] = m_fmacAcc[lane];
        }
        switch (fmac.source)
        {
        case FmacSource::Broadcast:
            for (uint32_t lane = 0; lane < 4u; ++lane)
                b[lane] = m_fmacVt[fmac.broadcast];
            break;
        case FmacSource::Q:
            for (uint32_t lane = 0; lane < 4u; ++lane)
                b[lane] = m_fmacQ;
            break;
        case FmacSource::I:
            for (uint32_t lane = 0; lane < 4u; ++lane)
                b[lane] = m_fmacI;
            break;
        case FmacSource::Full:
            for (uint32_t lane = 0; lane < 4u; ++lane)
                b[lane] = m_fmacVt[lane];
            break;
        case FmacSource::Cross:
            for (uint32_t lane = 0; lane < 4u; ++lane)
                b[lane] = m_fmacVt[crossRight[lane]];
            break;
        }
    };

    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        if (productSum)
        {
            // Every product condition accumulates into the sticky flags (the
            // current flags come from the sum). A float product is exact in
            // double.
            gatherOperands();
            float ignored = 0.0f;
            productSticky |= normalizeFmacExactResult(ignored, a[component] * b[component]) & 0xFu;
        }

        // With a single rounding toward zero, a normal result below FLT_MAX
        // means the exact result is nonzero, same-signed and in range: the
        // exact check below would only report the sign and keep the value.
        if (single)
        {
            uint32_t bits = 0;
            std::memcpy(&bits, &result[component], sizeof(bits));
            const uint32_t exponent = (bits >> 23) & 0xFFu;
            if (exponent != 0u && exponent != 0xFFu && (bits & 0x7FFFFFFFu) != 0x7F7FFFFFu)
            {
                laneFlags[component] = (bits & 0x80000000u) != 0u ? 0x2u : 0u;
                continue;
            }
            // Operands are zero or normal, so a sum or difference is a
            // multiple of the smallest denormal: one that rounds to zero is
            // exactly zero, signed by the same IEEE rule as the exact sum.
            if ((bits & 0x7FFFFFFFu) == 0u && (fmac.op == FmacOp::Add || fmac.op == FmacOp::Sub))
            {
                laneFlags[component] = (bits & 0x80000000u) != 0u ? 0x3u : 0x1u;
                continue;
            }
        }

        if (fmac.op == FmacOp::None)
        {
            uint32_t flags = 0u;
            result[component] = normalizeResult(result[component], flags);
            laneFlags[component] = static_cast<uint8_t>(flags);
            continue;
        }

        // The exact result in double classifies like the long double one: a
        // float product is exact in double, and zero, sign and FLT_MIN are
        // preserved by rounding toward zero. Only a sum that rounds to exactly
        // FLT_MAX may hide an overflow, so that case takes the long double.
        gatherOperands();
        const double product = a[component] * b[component];
        double exactResult = 0.0;
        switch (fmac.op)
        {
        case FmacOp::Add: exactResult = a[component] + b[component]; break;
        case FmacOp::Sub: exactResult = a[component] - b[component]; break;
        case FmacOp::Mul: exactResult = product; break;
        case FmacOp::MAdd: exactResult = acc[component] + product; break;
        case FmacOp::MSub: exactResult = acc[component] - product; break;
        case FmacOp::OuterMul: exactResult = component == 3u ? 0.0 : product; break;
        case FmacOp::OuterMSub: exactResult = component == 3u ? 0.0 : acc[component] - product; break;
        case FmacOp::None: break;
        }
        if (std::fabs(exactResult) == static_cast<double>(std::numeric_limits<float>::max()))
        {
            long double preciseResult = 0.0L;
            calculateFmacExactResult(component, preciseResult);
            laneFlags[component] = normalizeFmacExactResult(result[component], preciseResult);
        }
        else
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
        }
    }
}

template <typename Real>
bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               Real &result) const
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<Real>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const Real q = operand(m_state.q);
    const Real i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? Real(0)
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? Real(0)
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

template <typename Real>
uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  Real exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const Real magnitude = std::fabs(exactResult);
    const Real maximum = static_cast<Real>(std::numeric_limits<float>::max());
    const Real minimum = static_cast<Real>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == Real(0))
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    FlagPipelineEntry *const entry = pushFlags();
    if (!entry)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return;
    }
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    uint32_t productSticky = 0u;
    normalizeFmacResult(result, dest, laneFlags, productSticky);
    updateFmacFlags(laneFlags, dest, productSticky);
    applyDest(dst, result, dest);
}

void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    uint8_t laneFlags[4]{};
    uint32_t productSticky = 0u;
    normalizeFmacResult(result, dest, laneFlags, productSticky);
    updateFmacFlags(laneFlags, dest, productSticky);
    applyDestAcc(result, dest);
}

void VU1Interpreter::queueFsset(uint16_t immediate)
{
    for (uint32_t index = 0; index < m_flagCount; ++index)
    {
        FlagPipelineEntry &entry = m_flagPipeline[(m_flagHead + index) % kMaxFlagEntries];
        if (entry.issueCycle == m_cycle)
            entry.writesStatus = false;
    }

    if (FlagPipelineEntry *const entry = pushFlags())
    {
        entry->status = static_cast<uint32_t>(immediate) & 0xFC0u;
        entry->writesSticky = true;
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFEu);
}

void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    if (FlagPipelineEntry *const entry = pushFlags())
    {
        entry->clip = m_workingClip;
        entry->writesClip = true;
        return;
    }
    reportReservedInstruction(true, 0xFFFFFFFDu);
}

void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    for (uint32_t index = 0; index < m_flagCount; ++index)
    {
        FlagPipelineEntry &entry = m_flagPipeline[(m_flagHead + index) % kMaxFlagEntries];
        if (entry.issueCycle == m_cycle)
            entry.writesClip = false;
    }
    if (FlagPipelineEntry *const entry = pushFlags())
    {
        entry->clip = m_workingClip;
        entry->writesClip = true;
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFAu);
}

void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    noteReadyCycle(m_fdiv.readyCycle);
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    if (const int slot = claimSlot<2u>(m_efuMask); slot >= 0)
    {
        ScalarPipelineEntry &entry = m_efu[static_cast<uint32_t>(slot)];
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            noteReadyCycle(entry.readyCycle);
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

void VU1Interpreter::storeWords(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    if (address / 16u == g_traceStoreQword)
    {
        uint32_t lower = 0u;
        uint32_t upper = 0u;
        if (m_cachedVuCode && m_state.pc + 8u <= m_cachedCodeSize)
        {
            std::memcpy(&lower, m_cachedVuCode + m_state.pc, sizeof(lower));
            std::memcpy(&upper, m_cachedVuCode + m_state.pc + sizeof(lower),
                        sizeof(upper));
        }
        std::fprintf(stderr,
                     "[VU1 store] cycle=%llu pc=0x%x upper=0x%x lower=0x%x "
                     "qword=0x%x mask=0x%x "
                     "words={0x%x,0x%x,0x%x,0x%x}\n",
                     static_cast<unsigned long long>(m_cycle), m_state.pc,
                     upper, lower, address / 16u, laneMask, words[0],
                     words[1], words[2], words[3]);
    }
    // A store lands on the next cycle boundary, which run() reaches before
    // anything else can look at data memory, so write it straight away.
    if (m_activeVuData && address + 16u <= m_activeVuDataSize)
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((laneMask & laneForComponent(component)) != 0u)
                std::memcpy(m_activeVuData + address + component * 4u, &words[component], sizeof(uint32_t));
        }
    }
}

VU_FORCE_INLINE void VU1Interpreter::commitReadyPipelines()
{
    if (m_cycle >= m_nextCommitCycle)
        commitDuePipelines();
}

void VU1Interpreter::commitDuePipelines()
{
    uint64_t next = std::numeric_limits<uint64_t>::max();

    if (m_fdiv.valid)
    {
        if (m_fdiv.readyCycle <= m_cycle)
        {
            // FSSET replaces the D/I sticky bits, so order against it matters.
            commitFlags();
            m_state.q = m_fdiv.value;
            const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
            m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            m_fdiv = {};
        }
        else
        {
            next = std::min(next, m_fdiv.readyCycle);
        }
    }

    for (uint32_t bits = m_efuMask; bits != 0u;)
    {
        const int slot = nextSlot(bits);
        ScalarPipelineEntry &entry = m_efu[static_cast<uint32_t>(slot)];
        if (entry.readyCycle > m_cycle)
        {
            next = std::min(next, entry.readyCycle);
            continue;
        }
        m_state.p = entry.value;
        entry = {};
        m_efuMask &= ~(1u << slot);
    }

    m_nextCommitCycle = next;
}

void VU1Interpreter::progressXgkick()
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    ++m_xgkick.cycleCredit;
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        const uint32_t qwordSource = (m_xgkick.sourceAddress + m_xgkick.copiedBytes) % m_activeVuDataSize;
        if (qwordSource + 16u <= m_activeVuDataSize)
        {
            std::memcpy(m_xgkick.packet.data() + qwordOffset, m_activeVuData + qwordSource, 16u);
        }
        else
        {
            for (uint32_t i = 0; i < 16u; ++i)
            {
                const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes + i) % m_activeVuDataSize;
                m_xgkick.packet[m_xgkick.copiedBytes + i] = m_activeVuData[source];
            }
        }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else if (format == 2u || format == 3u)
                tagBytes += static_cast<uint64_t>(nloop) * 16u;
            else
            {
                reportReservedInstruction(false, 0xFFFFFFF8u);
                m_xgkick.active = false;
                return;
            }

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
    m_xgkick.active = false;
}

void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.totalBytes = 0;
    m_xgkick.copiedBytes = 0;
    m_xgkick.currentTagEnd = 0;
    m_xgkick.currentTagEop = false;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
}

VU_FORCE_INLINE void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    commitReadyPipelines();
    if (m_xgkick.active)
    {
        // PATH1 moves one qword every second cycle.
        if (m_xgkick.cycleCredit == 0u)
            m_xgkick.cycleCredit = 1u;
        else
            progressXgkick();
    }
}

void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
    {
        // With no XGKICK in flight a cycle does nothing until the next
        // pipeline commit, so skip straight to the cycle before it.
        if (!m_xgkick.active)
        {
            const uint64_t next = std::min(targetCycle, m_nextCommitCycle);
            if (next - 1u > m_cycle)
                m_cycle = next - 1u;
        }
        advanceOneCycle();
    }
}

bool VU1Interpreter::pipelinesPending() const
{
    return m_xgkick.active || m_flagCount != 0u ||
           m_nextCommitCycle != std::numeric_limits<uint64_t>::max() ||
           m_writebackHorizon > m_cycle;
}

void VU1Interpreter::flushPipelines()
{
    for (;;)
    {
        commitFlags();
        if (!pipelinesPending())
            break;
        if (!m_xgkick.active)
        {
            uint64_t next = m_nextCommitCycle;
            if (m_flagCount != 0u)
                next = std::min(next, m_flagPipeline[m_flagHead].readyCycle);
            if (m_writebackHorizon > m_cycle)
                next = std::min(next, m_writebackHorizon);
            if (next > m_cycle + 1u)
                m_cycle = next - 1u;
        }
        advanceOneCycle();
    }
}

VU_FORCE_INLINE uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    const __m128i reads = _mm_loadu_si128(reinterpret_cast<const __m128i *>(decoded.vfReads.data()));
    const __m128i newest = _mm_loadu_si128(reinterpret_cast<const __m128i *>(m_issued[0].vf.data()));
    const __m128i middle = _mm_loadu_si128(reinterpret_cast<const __m128i *>(m_issued[1].vf.data()));
    const __m128i oldest = _mm_loadu_si128(reinterpret_cast<const __m128i *>(m_issued[2].vf.data()));
    const uint32_t viReads = decoded.viReads;

    uint64_t ready = m_cycle;
    if (!_mm_testz_si128(reads, _mm_or_si128(_mm_or_si128(newest, middle), oldest)) ||
        (viReads & (m_issued[0].vi | m_issued[1].vi | m_issued[2].vi)) != 0u)
    {
        // Later pairs issue later, so the newest conflict is the one to wait for.
        uint64_t blocked = m_issued[2].readyCycle;
        if (!_mm_testz_si128(reads, middle) || (viReads & m_issued[1].vi) != 0u)
            blocked = m_issued[1].readyCycle;
        if (!_mm_testz_si128(reads, newest) || (viReads & m_issued[0].vi) != 0u)
            blocked = m_issued[0].readyCycle;
        ready = std::max(ready, blocked);
    }
    if (!decoded.hasResourceWait)
        return ready;
    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

VU_FORCE_INLINE void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    // A faster write to the same VI supersedes a slow one still pending.
    const uint32_t keep = ~static_cast<uint32_t>(decoded.viWrites);
    m_issued[2] = m_issued[1];
    m_issued[2].vi &= keep;
    m_issued[1] = m_issued[0];
    m_issued[1].vi &= keep;
    m_issued[0].vf = decoded.vfWrites;
    m_issued[0].vi = decoded.viSlowWrites;
    m_issued[0].readyCycle = m_cycle + kFmacLatency;
    m_writebackHorizon = std::max(m_writebackHorizon, m_cycle + decoded.maxWriteLatency);
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper) const
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        if (op == 0x08u || op == 0x09u || op == 0x0Au || op == 0x0Bu ||
            op == 0x0Cu || op == 0x0Du || op == 0x0Eu || op == 0x0Fu ||
            op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
            op == 0x29u || op == 0x2Du || op == 0x2Eu)
        {
            usage.accRead = dest;
        }
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
            usage.accWrite = dest;
            if ((special >= 0x08u && special <= 0x0Fu) ||
                special == 0x21u || special == 0x23u || special == 0x25u ||
                special == 0x27u || special == 0x29u || special == 0x2Du)
            {
                usage.accRead = dest;
            }
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special != 0x2Fu && special != 0x30u)
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(uint32_t lower) const
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x77:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7C:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x79:
            usage.latency = 29u;
            break;
        case 0x7D:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(decoded.lower);

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }

    const auto addLanes = [](std::array<uint64_t, 2> &set, uint32_t reg, uint32_t lanes)
    {
        const uint32_t bit = reg * 4u;
        set[bit / 64u] |= static_cast<uint64_t>(kReverseLanes[lanes & 0xFu]) << (bit % 64u);
    };
    for (const InstructionUsage *usage : {&decoded.upperUsage, &decoded.lowerUsage})
    {
        for (uint32_t index = 0; index < usage->vfReadCount; ++index)
        {
            const VfAccess &access = usage->vfRead[index];
            if (access.reg != 0u) // VF0 is never written, so never pending.
                addLanes(decoded.vfReads, access.reg, access.lanes);
        }
        decoded.viReads |= usage->viRead & 0xFFFEu;
    }

    const auto writeLatency = [&decoded](uint32_t latency)
    {
        decoded.maxWriteLatency = std::max(decoded.maxWriteLatency, static_cast<uint8_t>(latency));
    };
    // Every VF write has the FMAC latency.
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u && lowerWrite.lanes != 0u && decoded.suppressedLowerVf != lowerWrite.reg)
    {
        addLanes(decoded.vfWrites, lowerWrite.reg, lowerWrite.lanes);
        writeLatency(kFmacLatency);
    }
    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u && upperWrite.lanes != 0u)
    {
        addLanes(decoded.vfWrites, upperWrite.reg, upperWrite.lanes);
        writeLatency(kFmacLatency);
    }
    decoded.viWrites = decoded.lowerUsage.viWrite & 0xFFFEu;
    if (decoded.viWrites != 0u)
    {
        const uint32_t viLatency = decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency
                                                                      : decoded.lowerUsage.latency;
        if (viLatency == kFmacLatency)
            decoded.viSlowWrites = decoded.viWrites;
        writeLatency(viLatency);
    }
    if (decoded.upperUsage.accWrite != 0u)
        writeLatency(kAccForwardLatency);

    if ((decoded.lowerUsage.viWrite & 0xFFFEu) != 0u)
        decoded.firstViWrite = static_cast<uint8_t>(std::countr_zero(decoded.lowerUsage.viWrite & 0xFFFEu));

    const InstructionUsage &lower = decoded.lowerUsage;
    decoded.hasResourceWait = lower.pipeline == PipelineFdiv || lower.pipeline == PipelineEfu ||
                              lower.pipeline == PipelineXgkick || lower.waitQ || lower.waitP;
    decoded.plain = !decoded.iBit && decoded.upperVfShadowReg == 0u &&
                    !decoded.upperUsage.reserved && !lower.reserved;
    decoded.mayEnd = decoded.eBit || decoded.dBit || decoded.tBit;

    const uint32_t upperWord = decoded.upper;
    const uint32_t op = upperWord & 0x3Fu;
    const bool special = op >= 0x3Cu;
    const uint32_t selector = special ? ((upperWord & 3u) | ((upperWord >> 4) & 0x7Cu)) : op;

    UpperInstruction &fastUpper = decoded.upperOp;
    fastUpper.fs = FS(upperWord);
    fastUpper.ft = FT(upperWord);
    fastUpper.destination = special ? fastUpper.ft : FD(upperWord);
    fastUpper.broadcast = static_cast<uint8_t>(selector & 3u);
    fastUpper.lanes = kReverseLanes[DEST(upperWord)];
    if (special && (selector == 0x2Fu || selector == 0x30u))
    {
        fastUpper.kind = UpperKind::Nop;
    }
    else if (!decoded.upperUsage.reserved)
    {
        // decodeFmac does not know MAX, MINI and the conversions, so match those first.
        if (special && selector >= 0x10u && selector <= 0x17u)
            fastUpper.kind = selector < 0x14u ? UpperKind::IntToFloat : UpperKind::FloatToInt;
        else if (special && selector == 0x1Du)
            fastUpper.kind = UpperKind::Abs;
        else if (special && selector == 0x1Fu)
            fastUpper.kind = UpperKind::Clip;
        else if (!special && selector >= 0x10u && selector <= 0x13u)
            fastUpper.kind = UpperKind::MaxBroadcast;
        else if (!special && selector >= 0x14u && selector <= 0x17u)
            fastUpper.kind = UpperKind::MiniBroadcast;
        else if (!special && selector == 0x1Du)
            fastUpper.kind = UpperKind::MaxI;
        else if (!special && selector == 0x1Fu)
            fastUpper.kind = UpperKind::MiniI;
        else if (!special && selector == 0x2Bu)
            fastUpper.kind = UpperKind::Max;
        else if (!special && selector == 0x2Fu)
            fastUpper.kind = UpperKind::Mini;
        else if (const FmacDecode fmac = decodeFmac(upperWord); fmac.op != FmacOp::None)
            fastUpper.kind = UpperKind::fmac(fmac.op, fmac.source, special);
    }

    LowerInstruction &fastLower = decoded.lowerOp;
    const uint32_t word = decoded.lower;
    if (decoded.iBit || word == 0u || word == 0x8000033Cu)
    {
        fastLower.kind = LowerKind::Nop;
    }
    else if (!lower.reserved)
    {
        const uint32_t primary = (word >> 25) & 0x7Fu;
        const uint8_t dest = DEST(word);
        // ILW takes the first selected component, w when none is.
        const uint8_t component = (dest & 8u) ? 0u : (dest & 4u) ? 1u : (dest & 2u) ? 2u : 3u;
        fastLower.is = VIS(word);
        fastLower.it = VIT(word);
        fastLower.id = VID(word);
        fastLower.lanes = kReverseLanes[dest];
        fastLower.immediate = IMM11(word);
        decoded.readsFlags = primary >= 0x10u && primary <= 0x1Cu;

        switch (primary)
        {
        case 0x00:
            fastLower.kind = LowerKind::Lq;
            fastLower.it = FT(word);
            break;
        case 0x01:
            fastLower.kind = LowerKind::Sq;
            fastLower.is = VIT(word);
            fastLower.it = FS(word);
            break;
        case 0x04:
            fastLower.lanes = component;
            if (fastLower.it != 0u)
                fastLower.kind = LowerKind::Ilw;
            break;
        case 0x05:
            fastLower.kind = LowerKind::Isw;
            break;
        case 0x08:
        case 0x09:
        {
            const uint32_t imm15 = (word & 0x7FFu) | ((word >> 10) & 0x7800u);
            fastLower.immediate = static_cast<int16_t>(primary == 0x08u ? imm15 : 0u - imm15);
            if (fastLower.it != 0u)
                fastLower.kind = LowerKind::AddImmediate;
            break;
        }
        case 0x12:
            fastLower.kind = LowerKind::ClipAnd;
            break;
        case 0x20:
            fastLower.kind = LowerKind::Branch;
            break;
        case 0x28:
            fastLower.kind = LowerKind::BranchEqual;
            break;
        case 0x29:
            fastLower.kind = LowerKind::BranchNotEqual;
            break;
        case 0x40:
        {
            const uint32_t function = word & 0x3Fu;
            if (function == 0x30u && fastLower.id != 0u)
                fastLower.kind = LowerKind::Iadd;
            else if (function == 0x34u && fastLower.id != 0u)
                fastLower.kind = LowerKind::Iand;
            else if (function == 0x35u && fastLower.id != 0u)
                fastLower.kind = LowerKind::Ior;
            if (function < 0x3Cu)
                break;

            switch ((word & 3u) | ((word >> 4) & 0x7Cu))
            {
            case 0x30:
            case 0x31:
                fastLower.kind = ((word & 3u) | ((word >> 4) & 0x7Cu)) == 0x30u ? LowerKind::Move : LowerKind::Mr32;
                fastLower.is = FS(word);
                fastLower.it = FT(word);
                break;
            case 0x34:
                fastLower.kind = LowerKind::Lqi;
                fastLower.it = FT(word);
                break;
            case 0x35:
                fastLower.kind = LowerKind::Sqi;
                fastLower.is = VIT(word);
                fastLower.it = FS(word);
                break;
            case 0x3C:
                fastLower.is = FS(word);
                fastLower.lanes = static_cast<uint8_t>((word >> 21) & 3u);
                if (fastLower.it != 0u)
                    fastLower.kind = LowerKind::Mtir;
                break;
            default:
                break;
            }
            break;
        }
        default:
            break;
        }
    }
    return decoded;
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    const uint32_t pairCount = std::min<uint32_t>(codeSize / 8u, kMaxDecodedPairs);
    // MPG usually replaces a small part of micro memory.
    const bool canReuse = m_decodedCodeCacheValid && m_cachedCodeSize == codeSize;
    for (uint32_t i = 0; i < pairCount; ++i)
    {
        DecodedInstructionPair &cached = m_decodedCodeCache[i];
        uint32_t words[2];
        std::memcpy(words, vuCode + i * 8u, sizeof(words));
        if (!canReuse || cached.lower != words[0] || cached.upper != words[1])
            cached = decodeInstructionPair(vuCode, i * 8u);
    }

    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;
}

const VU1Interpreter::DecodedInstructionPair &VU1Interpreter::getDecodedInstructionPairForPc(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
{
    if ((pc & 7u) != 0u)
        return m_uncachedDecoded = decodeInstructionPair(vuCode, pc);

    const bool trackedVu1Code = memory != nullptr &&
                                ((m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                                 (m_unit == Unit::VU0 && vuCode == memory->getVU0Code()));
    if (!trackedVu1Code)
        return m_uncachedDecoded = decodeInstructionPair(vuCode, pc);

    const uint64_t generation = m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration();
    if (!m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize ||
        m_cachedCodeGeneration != generation)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }
    const uint32_t pairIndex = pc / 8u;
    if (pairIndex >= kMaxDecodedPairs)
        return m_uncachedDecoded = decodeInstructionPair(vuCode, pc);
    return m_decodedCodeCache[pairIndex];
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    m_stopRequested = true;
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    resetScheduler();
    m_state.pc = startPC & microAddressMask();
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    const int previousRoundingMode = std::fegetround();
    const bool useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    const uint64_t budgetEnd = m_cycle + maxCycles;
    bool programEnded = false;
    // Micro memory cannot change while a program runs, so validate the decode
    // cache on the first fetch only.
    const DecodedInstructionPair *program = nullptr;
    const bool tracing = g_traceVfRegister != kTraceDisabled || g_traceViRegister != kTraceDisabled;
    while (m_cycle < budgetEnd && !m_stopRequested)
    {
        commitReadyPipelines();
        if (m_state.pc + 8u > codeSize)
            break;

        const uint32_t pairIndex = m_state.pc / 8u;
        const bool cacheable = (m_state.pc & 7u) == 0u && pairIndex < kMaxDecodedPairs;
        const DecodedInstructionPair *pair;
        if (program && cacheable)
        {
            pair = &program[pairIndex];
        }
        else
        {
            pair = &getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
            if (pair != &m_uncachedDecoded)
                program = m_decodedCodeCache.data();
        }
        const DecodedInstructionPair &decoded = *pair;
        if (!decoded.plain && (decoded.upperUsage.reserved || decoded.lowerUsage.reserved))
        {
            reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
            break;
        }

        uint64_t readyCycle = calculatePairReadyCycle(decoded);
        while (readyCycle > m_cycle)
        {
            if (readyCycle >= budgetEnd)
            {
                advanceTo(budgetEnd);
                break;
            }
            advanceTo(readyCycle);
            readyCycle = calculatePairReadyCycle(decoded);
        }
        if (m_cycle >= budgetEnd)
            break;

        const uint8_t writtenVi = decoded.firstViWrite;
        const int32_t oldVi = writtenVi != 0u ? m_state.vi[writtenVi] : 0;

        if (decoded.plain) [[likely]]
        {
            execUpperDecoded(decoded);
            execLowerDecoded(decoded, vuData, dataSize, gs, memory);
        }
        else if (decoded.iBit)
        {
            execUpperDecoded(decoded);
            float immediate = 0.0f;
            std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
            m_state.i = normalizeOperand(immediate);
        }
        else
        {
            float oldVf[4]{};
            float upperVf[4]{};
            std::memcpy(oldVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(oldVf));
            execUpperDecoded(decoded);
            std::memcpy(upperVf,
                        m_state.vf[decoded.upperVfShadowReg],
                        sizeof(upperVf));
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        oldVf,
                        sizeof(oldVf));
            execLowerDecoded(decoded, vuData, dataSize, gs, memory);
            std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                        upperVf,
                        sizeof(upperVf));
        }

        if (tracing && g_traceVfRegister != kTraceDisabled)
        {
            auto traceVfWrite = [&](const VfAccess &write, const InstructionUsage &usage)
            {
                if (write.reg != g_traceVfRegister)
                    return;
                uint32_t words[4]{};
                std::memcpy(words, m_state.vf[write.reg], sizeof(words));
                const uint32_t latency = usage.vfLatency != 0u ? usage.vfLatency : usage.latency;
                std::fprintf(stderr,
                             "[VU1 VF write] cycle=%llu pc=0x%x upper=0x%x lower=0x%x "
                             "reg=%u mask=0x%x latency=%u "
                             "words={0x%x,0x%x,0x%x,0x%x} "
                             "vi={0x%x,0x%x,0x%x,0x%x,0x%x,0x%x,0x%x,0x%x,"
                             "0x%x,0x%x,0x%x,0x%x,0x%x,0x%x,0x%x,0x%x}\n",
                             static_cast<unsigned long long>(m_cycle), m_state.pc,
                             decoded.upper, decoded.lower, write.reg, write.lanes,
                             latency, words[0], words[1], words[2], words[3],
                             m_state.vi[0], m_state.vi[1], m_state.vi[2],
                             m_state.vi[3], m_state.vi[4], m_state.vi[5],
                             m_state.vi[6], m_state.vi[7], m_state.vi[8],
                             m_state.vi[9], m_state.vi[10], m_state.vi[11],
                             m_state.vi[12], m_state.vi[13], m_state.vi[14],
                             m_state.vi[15]);
            };
            traceVfWrite(decoded.upperUsage.vfWrite, decoded.upperUsage);
            if (decoded.lowerUsage.vfWrite.reg != decoded.suppressedLowerVf)
                traceVfWrite(decoded.lowerUsage.vfWrite, decoded.lowerUsage);
        }
        if (tracing && writtenVi == g_traceViRegister)
        {
            const uint32_t latency = decoded.lowerUsage.viLatency != 0u
                ? decoded.lowerUsage.viLatency
                : decoded.lowerUsage.latency;
            std::fprintf(stderr,
                         "[VU1 VI write] cycle=%llu pc=0x%x upper=0x%x lower=0x%x "
                         "reg=%u latency=%u value=0x%x\n",
                         static_cast<unsigned long long>(m_cycle), m_state.pc,
                         decoded.upper, decoded.lower, writtenVi, latency,
                         static_cast<uint32_t>(m_state.vi[writtenVi]));
        }

        m_viBranchBackupValid = false;

        // VF, VI and ACC results are written back immediately: these
        // registers are interlocked, so calculatePairReadyCycle stalls every
        // reader until the result's latency has elapsed and an early value
        // can never be observed. (Flags, Q, P and stores are not interlocked
        // and keep their pipelines.)
        markPairWrites(decoded);
        if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
            recordViWriteForBranch(writtenVi, oldVi);

        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        m_state.vi[0] = 0;

        uint32_t nextPc = m_state.pc + 8u;
        if (nextPc >= codeSize)
            nextPc = 0u;
        m_state.pc = nextPc;

        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0u)
            {
                m_state.pc = m_state.branchTarget & microAddressMask();
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        if (!decoded.mayEnd && !m_state.ebit && !m_state.haltAfterDelaySlot) [[likely]]
        {
            advanceOneCycle();
            continue;
        }

        const bool dHalt = decoded.dBit && m_state.dBitEnabled;
        const bool tHalt = decoded.tBit && m_state.tBitEnabled;
        const bool haltBit = dHalt || tHalt;
        const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

        if (m_state.haltAfterDelaySlot)
        {
            m_state.stoppedByD = m_pendingHaltD;
            m_state.stoppedByT = m_pendingHaltT;
            programEnded = true;
        }
        else if (m_state.ebit)
            programEnded = true;
        else if (haltBit && !haltBranch)
        {
            m_state.stoppedByD = dHalt;
            m_state.stoppedByT = tHalt;
            programEnded = true;
        }
        else if (decoded.eBit)
            m_state.ebit = true;
        else if (haltBranch)
        {
            m_state.haltAfterDelaySlot = true;
            m_pendingHaltD = dHalt;
            m_pendingHaltT = tHalt;
        }

        advanceOneCycle();
        if (programEnded)
            break;
    }

    if (programEnded)
    {
        flushPipelines();
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    commitFlags();
    m_state.cycles = m_cycle;
    if (useVuRounding && previousRoundingMode != -1)
        std::fesetround(previousRoundingMode);
}
