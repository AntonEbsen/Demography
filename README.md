# Historical demography: research code and coursework

Anton Meier Ebsen Jørgensen · MSc student in Economics, University of Copenhagen

## Bismarck’s Lost Battle

**Marriage, Fertility, and the Kulturkampf in Prussia, 1867–1890** is a completed demography course paper submitted on 29 May 2026, not a peer-reviewed publication.

- [Paper and project overview](https://www.antonebsen.com/en/projects/demography-project)
- [Submitted paper (PDF)](https://www.antonebsen.com/en/projects/demography-project.pdf)
- [Research code and setup](exam_project2/README.md)
- [Figure 4.1 validation: results and discrepancy](exam_project2/reports/figure_4_1_validation.md)
- [One-command Figure 4.1 reproduction (Python 3.12)](reproduction/figure-4-1/README.md)
- [Data appendix for the evolving analysis](exam_project2/DATA_APPENDIX.md)

The repository also contains later exploratory extensions, including longer time windows and additional methods. Current notebooks, the digital monograph, and cached outputs are **not a frozen replication package for the submitted paper**. The Figure 4.1 check validates one descriptive figure and related sample counts, not the regressions or causal interpretation.

## Repository guide

| Directory | Contents |
|---|---|
| `exam_project2/` | Data loaders, analysis modules, notebooks, and validation |
| `kulturkampf_site/` | Digital monograph source; may reflect later extensions |
| `tests/` | Software checks; see the test README for scope |
| `lectures/` | Course materials and notes |
| `translation_of_STATA_code/` | Python translations of econometric examples |
| `archived/` | Earlier coursework, including the Factory Acts project |

Third-party data retain their source conditions; the software license does not relicense them. See [data provenance](exam_project2/data/README.md).

## AI assistance and responsibility

Earlier repository professionalization used the AI coding agent **Antigravity** for the monograph architecture, automation, package structure, Binder configuration, and technical documentation. That acknowledgment is retained; configuration alone is not evidence that every workflow currently works. The author supplied the research specifications, historical interpretation, and domain expertise, and remains responsible for the work.

The Figure 4.1 validation code and documentation cleanup were developed with Codex assistance. Automated tests are not evidence of causal validity.

## Contact

[Personal website](https://www.antonebsen.com/en) · [LinkedIn](https://www.linkedin.com/in/antonebsen/) · anton.ebsen@gmail.com
