import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_DATA_YEAR = 2025
MIN_DAYS = 300


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 유효한 데이터만 사용
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# -----------------------------
# 연도별 평균기온 계산
# -----------------------------
yearly = (
    df[df["연도"] <= LAST_DATA_YEAR]
    .groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count"),
    )
    .reset_index()
)

# 관측일수가 300일 미만인 해 제외
yearly = yearly[yearly["관측일수"] >= MIN_DAYS].copy()

# 1908년부터 지난 연수를 독립변수로 사용
yearly["경과연수"] = yearly["연도"] - BASE_YEAR

# 정렬
yearly = yearly.sort_values("연도").reset_index(drop=True)


# -----------------------------
# 회귀분석
# -----------------------------
x = yearly["경과연수"].to_numpy()
y = yearly["평균기온"].to_numpy()

# y = intercept + slope * x
slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


def predict_temperature(year):
    """회귀식으로 특정 연도의 예상 평균기온 계산"""
    elapsed_years = year - BASE_YEAR
    return intercept + slope * elapsed_years


# -----------------------------
# 화면
# -----------------------------
st.title("🌡️ 서울 기온 예측기")

st.markdown(
    """
서울의 연평균기온 데이터를 이용해 선형회귀를 수행하고,
선택한 연도의 예상 평균기온을 확인할 수 있습니다.
"""
)

# 분석 기간 정보
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
n_years = len(yearly)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{n_years}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")


st.caption(
    f"분석 조건: {LAST_DATA_YEAR}년 이하이며 관측일수가 "
    f"{MIN_DAYS}일 이상인 연도만 사용"
)


# -----------------------------
# 연도 슬라이더
# -----------------------------
selected_year = st.slider(
    "예상 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

predicted_temp = predict_temperature(selected_year)

st.subheader(f"{selected_year}년 예상 평균기온")

st.markdown(
    f"""
    <div style="
        font-size: 3.5rem;
        font-weight: 700;
        color: #d62728;
        margin: 10px 0 25px 0;
    ">
        {predicted_temp:.2f} °C
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# 상관계수
# -----------------------------
st.subheader("상관계수")

st.metric(
    "연도(1908년부터의 경과연수)와 연평균기온의 상관계수",
    f"{correlation:.4f}",
)


# -----------------------------
# Plotly 그래프
# -----------------------------
# 회귀선은 1900~2100년 전체 범위에서 표시
line_years = np.arange(1900, 2101)
line_elapsed = line_years - BASE_YEAR
line_temperatures = intercept + slope * line_elapsed

fig = go.Figure()

# 실제 관측 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
            color="#1f77b4",
            opacity=0.75,
        ),
        customdata=yearly["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperatures,
        mode="lines",
        name="회귀 직선",
        line=dict(
            color="#d62728",
            width=3,
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예상기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 현재 선택한 연도의 예상값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name="선택 연도 예상값",
        marker=dict(
            size=14,
            color="#ff7f0e",
            line=dict(
                color="white",
                width=2,
            ),
        ),
        hovertemplate=(
            f"{selected_year}년<br>"
            "예상 평균기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis=dict(
        title="연도",
        # x축은 실제 연도를 그대로 표시
        range=[1900, 2100],
        dtick=10,
        tickformat="d",
    ),
    yaxis=dict(
        title="평균기온 (°C)",
    ),
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    height=600,
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# 회귀식
# -----------------------------
st.subheader("회귀식")

st.latex(
    rf"\hat{{T}} = {intercept:.4f} "
    rf"{slope:+.4f} \times (\mathrm{{연도}} - 1908)"
)

st.write(
    f"1908년부터의 경과연수를 독립 변수로 하여 "
    f"기울기 {slope:.4f} °C/년으로 회귀 직선을 계산했습니다."
)


# -----------------------------
# 사용된 연도 데이터
# -----------------------------
with st.expander("회귀에 사용된 연도별 데이터 보기"):
    display_df = yearly.copy()
    display_df["평균기온"] = display_df["평균기온"].round(2)

    st.dataframe(
        display_df[
            ["연도", "관측일수", "평균기온", "경과연수"]
        ],
        use_container_width=True,
        hide_index=True,
    )
