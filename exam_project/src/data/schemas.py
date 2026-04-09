import pandera as pa
from pandera.typing import Series

class PanelSchema(pa.DataFrameModel):
    """
    Schema for the master panel dataset used in the 'Cost of Quality' research.
    """
    REGDIST: Series[str] = pa.Field(description="Registration District Name")
    Year: Series[int] = pa.Field(isin=[1851, 1861, 1871, 1881], description="Census Year")
    
    # Core Demographic Variables
    TFR: Series[float] = pa.Field(ge=0, le=10, description="Total Fertility Rate")
    IMR: Series[float] = pa.Field(ge=0, description="Infant Mortality Rate")
    
    # Economic/Labor Variables
    F_TEX: Series[float] = pa.Field(ge=0, le=100, description="Share of females in textiles")
    F_CL_1013: Series[float] = pa.Field(ge=0, le=100, nullable=True, description="Share of female child labor (10-13)")
    
    class Config:
        strict = False  # Allow other columns but validate these core ones
        coerce = True
