"""QWK and recall of published DAiSEE systems, derived from their printed test confusion matrices.

The matrices are transcribed from the cited papers (rows = true level 0..3, columns = predicted
level 0..3, official DAiSEE test set of 1,784 clips with class counts 4 / 84 / 882 / 814):
  abedi2021resnettcn      Fig. 3(g)   ResNet + TCN
  abedi2024affectordinal  Table 6(d)  ordinal TCN on affect and behavioural features
  abedi2025supervised     Table IV(d) supervised contrastive + ordinal
ViBED-Net prints only a row-normalised matrix rounded to two decimals, so no exact value is derived.

Run from the repository root:  python analysis_v3/published_qwk.py
Output: paper/tables/published_qwk.tex (LaTeX macros) and the values on stdout.
"""
from pathlib import Path

MATRICES = {
    "ResTcn": [[0, 0, 4, 0], [0, 0, 64, 20], [0, 0, 616, 266], [0, 0, 290, 524]],
    "OrdTcn": [[1, 0, 3, 0], [1, 21, 48, 14], [10, 0, 669, 203], [0, 3, 300, 511]],
    "Contrastive": [[0, 1, 2, 1], [0, 29, 34, 21], [2, 13, 617, 250], [0, 18, 241, 555]],
}


def qwk(matrix):
    k = len(matrix)
    n = float(sum(sum(row) for row in matrix))
    rows = [sum(row) for row in matrix]
    cols = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    observed = 0.0
    expected = 0.0
    for i in range(k):
        for j in range(k):
            weight = (i - j) ** 2 / float((k - 1) ** 2)
            observed += weight * matrix[i][j] / n
            expected += weight * rows[i] * cols[j] / (n * n)
    return 1.0 - observed / expected


def main():
    lines = ["% written by analysis_v3/published_qwk.py -- do not edit"]
    for name, matrix in MATRICES.items():
        n = sum(sum(row) for row in matrix)
        accuracy = 100.0 * sum(matrix[i][i] for i in range(4)) / n
        recall_low = matrix[1][1] / float(sum(matrix[1]))
        value = qwk(matrix)
        print("%-12s n=%d acc=%.2f qwk=%.3f recall(level 1)=%.2f" % (name, n, accuracy, value, recall_low))
        lines.append("\\newcommand{\\pub%sQwk}{%.2f}" % (name, value))
        lines.append("\\newcommand{\\pub%sRecOne}{%.2f}" % (name, recall_low))
    first = MATRICES["ResTcn"]
    majority = 100.0 * max(sum(row) for row in first) / sum(sum(row) for row in first)
    lines.append("\\newcommand{\\pubMajority}{%.1f}" % majority)
    out = Path(__file__).resolve().parent.parent / "paper" / "tables" / "published_qwk.tex"
    out.write_text("\n".join(lines) + "\n")
    print("majority %.1f%%; wrote %s" % (majority, out))


if __name__ == "__main__":
    main()
