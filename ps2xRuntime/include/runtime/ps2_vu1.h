#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

class GS;
class PS2Memory;

struct VU1State
{
    float vf[32][4];
    int32_t vi[16];
    float acc[4];
    float q;
    float p;
    float i;
    uint32_t r;
    uint32_t pc;
    uint32_t mac;
    uint32_t clip;
    uint32_t status;
    uint64_t cycles;
    bool ebit;
    bool haltAfterDelaySlot;
    bool dBitEnabled;
    bool tBitEnabled;
    bool stoppedByD;
    bool stoppedByT;
    uint32_t top;  // VIF TOP visible to XTOP
    uint32_t itop; // VIF ITOP visible to XITOP

    bool branchPending;
    uint32_t branchTarget;
    uint32_t branchDelay;
};

struct VU1XgkickDebugState
{
    bool active = false;
    uint32_t sourceAddress = 0;
    uint32_t totalBytes = 0;
    uint32_t copiedBytes = 0;
    uint32_t currentTagEnd = 0;
    uint32_t cycleCredit = 0;
    uint64_t issueCycle = 0;
    bool currentTagEop = false;
};

class VU1Interpreter
{
public:
    enum class Unit : uint8_t
    {
        VU0,
        VU1
    };

    explicit VU1Interpreter(Unit unit = Unit::VU1);

    void reset();

    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t top = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t top = 0, uint32_t itop = 0, uint32_t maxCycles = 65536);

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }
    VU1XgkickDebugState xgkickDebugState() const;

private:
    enum Pipeline : uint8_t
    {
        PipelineNone = 0,
        PipelineFmac,
        PipelineLsu,
        PipelineFdiv,
        PipelineEfu,
        PipelineIalu,
        PipelineBranch,
        PipelineXgkick
    };

    struct VfAccess
    {
        uint8_t reg = 0;
        uint8_t lanes = 0;
    };

    struct InstructionUsage
    {
        std::array<VfAccess, 2> vfRead{};
        VfAccess vfWrite{};
        uint8_t vfReadCount = 0;
        uint16_t viRead = 0;
        uint16_t viWrite = 0;
        uint8_t accRead = 0;
        uint8_t accWrite = 0;
        uint8_t latency = 0;
        uint8_t vfLatency = 0;
        uint8_t viLatency = 0;
        Pipeline pipeline = PipelineNone;
        bool waitQ = false;
        bool waitP = false;
        bool readsClip = false;
        bool writesClip = false;
        bool delaysNextBranchRead = false;
        bool reserved = false;
    };

    // An upper instruction, decoded once when the micro memory changes.
    struct UpperInstruction
    {
        uint8_t kind = 0; // UpperKind
        uint8_t broadcast = 0;
        uint8_t fs = 0;
        uint8_t ft = 0;
        uint8_t destination = 0;
        uint8_t lanes = 0; // DEST with bit 0 = x, as movemask reports it.
    };

    // The same for the lower instructions that make up most programs.
    struct LowerInstruction
    {
        uint8_t kind = 0; // LowerKind
        uint8_t is = 0;
        uint8_t it = 0;
        uint8_t id = 0;
        uint8_t lanes = 0;
        int16_t immediate = 0;
    };

    struct DecodedInstructionPair
    {
        static constexpr uint8_t UpperVfShadowMask = 0x1Fu;
        static constexpr uint8_t Plain = 1u << 7;

        static constexpr uint8_t LowerFdiv = 1u << 0;
        static constexpr uint8_t LowerEfu = 1u << 1;
        static constexpr uint8_t LowerWaitQ = 1u << 2;
        static constexpr uint8_t LowerWaitP = 1u << 3;
        static constexpr uint8_t LowerXgkick = 1u << 4;
        static constexpr uint8_t LowerBranch = 1u << 5;
        static constexpr uint8_t DelaysNextBranchRead = 1u << 6;
        static constexpr uint8_t ResourceWaitMask =
            LowerFdiv | LowerEfu | LowerWaitQ | LowerWaitP | LowerXgkick;

        uint32_t lower = 0;
        uint32_t upper = 0;
        // Precomputed from the usages for the interlock check. A VF lane is bit
        // reg * 4 + lane of a 128-bit set and a VI register is its own bit.
        std::array<uint64_t, 2> vfReads{};
        std::array<uint64_t, 2> vfWrites{};
        UpperInstruction upperOp{};
        LowerInstruction lowerOp{};
        uint16_t viReads = 0;
        uint16_t viWrites = 0;
        uint16_t viSlowWrites = 0; // the ones that take kFmacLatency
        uint8_t executionInfo = 0;
        uint8_t firstViWrite = 0;
        uint8_t maxWriteLatency = 0;
        uint8_t lowerInfo = 0;

        uint8_t upperVfShadowReg() const
        {
            return static_cast<uint8_t>(executionInfo & UpperVfShadowMask);
        }
        bool isPlain() const { return (executionInfo & Plain) != 0u; }
        bool lowerHas(uint8_t flags) const { return (lowerInfo & flags) != 0u; }
    };
    static_assert(sizeof(DecodedInstructionPair) == 64u);

    struct FlagPipelineEntry
    {
        uint64_t readyCycle = 0;
        uint64_t issueCycle = 0;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t extraSticky = 0;
        uint32_t clip = 0;
        bool writesMac = false;
        bool writesStatus = false;
        bool writesSticky = false;
        bool writesClip = false;
    };

    struct ScalarPipelineEntry
    {
        uint64_t readyCycle = 0;
        float value = 0.0f;
        uint32_t statusDi = 0;
        bool valid = false;
    };

    struct XgkickPipeline
    {
        static constexpr uint32_t kBufferSize = 0x10000u;
        std::array<uint8_t, kBufferSize> packet{};
        uint32_t sourceAddress = 0;
        uint32_t totalBytes = 0;
        uint32_t copiedBytes = 0;
        uint32_t currentTagEnd = 0;
        uint32_t cycleCredit = 0;
        uint64_t issueCycle = 0;
        bool active = false;
        bool currentTagEop = false;
    };

    static constexpr uint32_t kFmacLatency = 4u;
    static constexpr uint32_t kAccForwardLatency = 1u;
    static constexpr uint32_t kMaxFlagEntries = 8u;
    static constexpr uint32_t kMaxDecodedPairs = 0x4000u / 8u;

    Unit m_unit;
    VU1State m_state;
    std::array<DecodedInstructionPair, kMaxDecodedPairs> m_decodedCodeCache{};
    const uint8_t *m_cachedVuCode = nullptr;
    const PS2Memory *m_cachedMemory = nullptr;
    uint32_t m_cachedCodeSize = 0;
    uint64_t m_cachedCodeGeneration = 0;
    bool m_decodedCodeCacheValid = false;

    std::array<FlagPipelineEntry, kMaxFlagEntries> m_flagPipeline{};
    ScalarPipelineEntry m_fdiv{};
    std::array<ScalarPipelineEntry, 2> m_efu{};
    XgkickPipeline m_xgkick{};
    // Only a write with the four-cycle latency can stall a later pair (one
    // cycle has passed before the next can issue), and only for the three
    // pairs after it. So the last three pairs' slow writes are the scoreboard.
    struct IssuedWrites
    {
        std::array<std::array<uint64_t, 2>, 3> vf{};
        std::array<uint64_t, 3> readyCycle{};
        std::array<uint32_t, 3> vi{};
    };
    IssuedWrites m_issued{}; // newest first
    std::array<uint64_t, 2> m_issuedVfAggregate{};
    uint32_t m_issuedViAggregate = 0;

    uint64_t m_cycle = 0;
    uint64_t m_efuResourceReady = 0;
    // Earliest readyCycle of any Q or P entry (max when none), so the per-cycle
    // commit and pending checks don't scan every pipeline.
    uint64_t m_nextCommitCycle = std::numeric_limits<uint64_t>::max();
    // Latest cycle at which an issued VF/VI/ACC write lands. The values are
    // written back at issue (see run), but a program only ends once its
    // writes would have committed.
    uint64_t m_writebackHorizon = 0;
    // Every flag write has the FMAC latency, so that pipeline is a FIFO. It is
    // committed lazily: before an instruction that reads flags, when it fills
    // up, and when the program stops.
    uint32_t m_flagHead = 0;
    uint32_t m_flagCount = 0;
    // Occupied slots per pipeline (bit i = entry i valid); commits walk set
    // bits in ascending slot order, the same order as a full array scan.
    uint32_t m_efuMask = 0;
    DecodedInstructionPair m_uncachedDecoded{};
    uint32_t m_workingClip = 0;
    uint32_t m_currentUpperInstruction = 0;
    // Normalized operands of the executing upper instruction (set by
    // execUpper, read by normalizeFmacResult).
    const float *m_fmacVs = nullptr;
    const float *m_fmacVt = nullptr;
    const float *m_fmacAcc = nullptr;
    float m_fmacQ = 0.0f;
    float m_fmacI = 0.0f;
    int32_t m_viBranchBackupValue = 0;
    uint8_t m_viBranchBackupReg = 0;
    bool m_viBranchBackupValid = false;
    uint8_t *m_activeVuData = nullptr;
    uint32_t m_activeVuDataSize = 0;
    GS *m_activeGs = nullptr;
    PS2Memory *m_activeMemory = nullptr;
    bool m_stopRequested = false;
    bool m_pendingHaltD = false;
    bool m_pendingHaltT = false;

    void run(uint8_t *vuCode, uint32_t codeSize,
             uint8_t *vuData, uint32_t dataSize,
             GS &gs, PS2Memory *memory, uint32_t maxCycles);

    InstructionUsage decodeUpperUsage(uint32_t upper) const;
    InstructionUsage decodeLowerUsage(uint32_t lower) const;
    static void addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static void addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static uint8_t vfReadLanes(const InstructionUsage &usage, uint8_t reg);
    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    const DecodedInstructionPair &getDecodedInstructionPairForPc(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc);
    void rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize, const PS2Memory *memory, uint64_t generation);

    void execUpper(uint32_t instr);
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    void applyFmacDest(float *dst, float *result, uint8_t dest);
    void applyFmacDestAcc(float *result, uint8_t dest);
    void normalizeFmacResult(float *result, uint8_t dest, uint8_t laneFlags[4], uint32_t &productSticky);
    template <typename Real>
    bool calculateFmacExactResult(uint32_t component, Real &result) const;
    template <typename Real>
    uint8_t normalizeFmacExactResult(float &value, Real exactResult) const;
    void updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest, uint32_t extraSticky);
    FlagPipelineEntry *pushFlags();
    void commitFlags();
    template <uint8_t Kind>
    void execFmac(const UpperInstruction &op, uint32_t instruction);
    void execUpperDecoded(const DecodedInstructionPair &decoded);
    void execLowerDecoded(const DecodedInstructionPair &decoded, uint8_t *vuData, uint32_t dataSize,
                          GS &gs, PS2Memory *memory);
    void queueFsset(uint16_t immediate);
    void queueClip(uint32_t clip);
    void queueFcset(uint32_t clip);
    void queueQ(float value, uint32_t latency, uint32_t statusDi);
    void queueP(float value, uint32_t latency);
    void storeWords(uint32_t address, const uint32_t words[4], uint8_t laneMask);
    void startXgkick(uint32_t qwordAddress);

    void resetScheduler();
    void commitReadyPipelines();
    void commitDuePipelines();
    void noteReadyCycle(uint64_t readyCycle)
    {
        if (readyCycle < m_nextCommitCycle)
            m_nextCommitCycle = readyCycle;
    }
    void advanceOneCycle();
    void advanceTo(uint64_t targetCycle);
    void flushPipelines();
    void progressXgkick();
    void finishXgkick();
    uint64_t calculatePairReadyCycle(const DecodedInstructionPair &decoded) const;
    void markPairWrites(const DecodedInstructionPair &decoded);
    bool pipelinesPending() const;

    // Denormals flush to signed zero; Inf/NaN clamp to the signed max.
    float normalizeOperand(float value) const
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent == 0u)
            bits &= 0x80000000u;
        else if (exponent == 0xFFu)
            bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    float normalizeResult(float value, uint32_t &laneFlags) const;
    uint32_t microAddressMask() const;
    int32_t readBranchVi(uint8_t reg) const;
    void recordViWriteForBranch(uint8_t reg, int32_t oldValue);
    void reportReservedInstruction(bool upper, uint32_t instruction);
    float broadcast(const float *vf, uint8_t bc);
};

#endif
