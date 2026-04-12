# Data Dictionary: The "Cost of Quality" Project

This dictionary defines the key variables used in the econometric analysis of the Victorian fertility transition (1851–1881). Data is primarily sourced from **Populations Past (Cambridge University)** and the **1831 Census of Great Britain**.

| Variable | Description | Source | Theoretical Role |
|:---|:---|:---|:---|
| **TFR** | Total Fertility Rate (Age-Standardized) | Populations Past | **Primary Outcome** |
| **Industrial_Ratio_1831** | Share of males (20+) in manufacturing | 1831 Census | **Treatment Dosage** |
| **F_CL_1013** | Share of girls (10-13) in the labor force | Populations Past | Child Quantity Mechanism |
| **F_CL_1418** | Share of girls (14-18) in the labor force | Populations Past | Teen Labor Substitution |
| **C_TEACHER** | Density of Teachers (per 10k pop) | Populations Past | Proxy for **Child Quality** |
| **IMR** | Infant Mortality Rate (Deaths < 1yr / 1000) | Populations Past | Biological / Replacement Control |
| **ECMR** | Early Childhood Mortality Rate | Populations Past | Extended mortality anchor |
| **F_SMAM** | Singulate Mean Age at Marriage (Females) | Populations Past | Postponement Mechanism |
| **TMFR** | Total Marital Fertility Rate | Populations Past | Within-marriage stopping |
| **HOUSE_SERV** | Share of pop employed as domestic servants | Populations Past | Proxy for Middle Class wealth |
| **SC1** | Social Class 1 (Higher Professional) share | populations Past | Upper-class norm anchor |
| **SC3** | Social Class 3 (Skilled Manual) share | Populations Past | Skill premium anchor |
| **F_CEL_4554** | Never-married share (Females 45-54) | Populations Past | Lifetime Celibacy proxy |
| **HH_KIN** | Density of extended kin in household | Populations Past | Household adaptation proxy |
| **IRL_F** | Share of females born in Ireland | Populations Past | Cultural/Religious control |
| **SR** | Sex Ratio (Males per 100 Females) | Populations Past | Marriage market density |
| **POP_DENS** | Population Density | Populations Past | Urbanization control |
| **ILEG_RATIO** | Illegitimacy share in births | Populations Past | Moral Geography proxy |

## Mathematical Definitions
- **TFR**: Sum of Age-Specific Fertility Rates (ASFR) multiplied by 5 (for 5-year brackets).
- **IMR**: (Deaths of children <1 year / Total Births) * 1000.
- **Industrial_Ratio_1831**: Normalized measure [0,1] at the Registration District level.
