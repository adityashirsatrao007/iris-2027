"""Convert the manuscript source to the Springer svmult proceedings class."""
from pathlib import Path
import re

src = Path("paper/main_lnncs_backup.tex").read_text(encoding="utf-8")
start = src.index(r"\begin{abstract}")
end = src.index(r"\end{abstract}", start)
abstract = src[start + len(r"\begin{abstract}"):end]
abstract = re.sub(r"\\keywords\{.*?\}", "", abstract, flags=re.S).strip()

pre, rest = src.split(r"\begin{document}", 1)
rest = rest[rest.index(r"\maketitle") + len(r"\maketitle"):]
rest = rest[rest.index(r"\section{Introduction}"):]
pre = re.sub(r"\\documentclass\[[^]]*\]\{llncs\}",
             lambda _: r"\documentclass[graybox]{svmult}", pre)
pre = pre.replace(r"\usepackage[T1]{fontenc}",
                  "\\usepackage[T1]{fontenc}\n"
                  "\\usepackage{mathptmx}\n"
                  "\\usepackage{helvet}\n"
                  "\\usepackage{courier}\n"
                  "\\usepackage{cite}")
pre = re.sub(r"\\title\{(.*?)\}\s*\\titlerunning\{.*?\}",
             lambda m: r"\title*{" + m.group(1) + "}", pre, flags=re.S)
pre = re.sub(r"\\author\{.*?\}", lambda _: r"\author{Aditya Vishal Shirsatrao}",
             pre, count=1, flags=re.S)
pre = re.sub(
    r"\\institute\{.*?\\email\{adityashirsatrao007@gmail.com\}\}",
    lambda _: r"\institute{N. K. Orchid College of Engineering \& Technology, Solapur, India\email{adityashirsatrao007@gmail.com}}",
    pre, count=1, flags=re.S)
pre = pre.replace(r"\usepackage{url}", r"\usepackage{url}")
rest = rest.replace(r"\bibliographystyle{splncs04}",
                    r"\bibliographystyle{spmpsci}")
out = pre + r"\begin{document}" + "\n"
out += r"\title*{Original, Restored, or Generated:\\Restoration as an Unmodelled Provenance State}" + "\n"
out += r"\author{Aditya Vishal Shirsatrao}" + "\n"
out += r"\institute{N. K. Orchid College of Engineering \& Technology, Solapur, India\email{adityashirsatrao007@gmail.com}}" + "\n"
out += r"\maketitle" + "\n"
out += r"\abstract*{" + abstract + "}\n"
out += r"\abstract{" + abstract + "}\n"
out += rest
Path("paper/main_svmult.tex").write_text(out, encoding="utf-8")
print("Wrote paper/main_svmult.tex")
