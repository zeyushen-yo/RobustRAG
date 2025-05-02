import re
from pathlib import Path

# -------------------------------------------------------------------
# Hard-coded log locations
# -------------------------------------------------------------------
FILE_A = Path("/home/zs7353/RobustRAG/slurm-63884842.out")   # baseline / first run
FILE_B = Path("/home/zs7353/RobustRAG/slurm-63884949.out")   # second run

# -------------------------------------------------------------------
ITEM_RE = re.compile(r"item:\s+(\d+)\s*====")   # captures the question ID


def parse_log(path: Path) -> dict[int, bool]:
    """
    Return {question_id: is_correct} for a single log.
    Marks a question as correct if at least one line *before* its
    header block contains the substring “correct!”.
    """
    status: dict[int, bool] = {}
    current_qid: int | None = None
    saw_correct = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "correct!" in line:
                saw_correct = True
                continue

            m = ITEM_RE.search(line)
            if m:
                # store result for the previous question
                if current_qid is not None:
                    status[current_qid] = saw_correct
                # start a new block
                current_qid = int(m.group(1))
                saw_correct = False

    # don’t forget the very last question
    if current_qid is not None:
        status[current_qid] = saw_correct
    return status


# -------------------------------------------------------------------
# Parse both logs and compute the diff
# -------------------------------------------------------------------
stats_a = parse_log(FILE_A)
stats_b = parse_log(FILE_B)

diff_ids = sorted(
    qid for qid, correct_a in stats_a.items()
    if not correct_a and stats_b.get(qid, False)
)

print(f"{len(diff_ids)} questions were missed by {FILE_A.name} "
      f"but answered correctly by {FILE_B.name}:")
print(diff_ids)