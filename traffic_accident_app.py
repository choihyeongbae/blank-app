import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="경북 어린이 교통사고 분석",
    page_icon="🚸",
    layout="wide"
)

# 커스텀 CSS - 글래스모피즘 스타일
st.markdown("""
    <style>
    /* 전체 배경 그라데이션 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 모든 텍스트 흰색 */
    .main * {
        color: white !important;
    }
    
    /* 제목 스타일 */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 2rem !important;
    }
    
    h2, h3 {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* 카드 스타일 (글래스모피즘) */
    .stMetric, .element-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    /* 메트릭 카드 */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem !important;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* 셀렉트박스 스타일 */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* 데이터프레임 스타일 */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 10px;
    }
    
    /* 차트 배경 */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 구분선 */
    hr {
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.3);
        margin: 2rem 0;
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white !important;
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        # 여기에 실제 데이터 파일 경로를 입력하세요
        df = pd.read_csv('경북_어린이교통사고_2019_2023.csv', encoding='utf-8-sig')
        return df
    except:
        # 샘플 데이터 생성
        st.warning("⚠️ 데이터 파일을 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
        data = {
            '시군구': ['포항시'] * 50 + ['경주시'] * 40 + ['안동시'] * 30,
            '사고건수': np.random.randint(1, 20, 120),
            '사망자수': np.random.randint(0, 3, 120),
            '중상자수': np.random.randint(0, 5, 120),
            '경상자수': np.random.randint(0, 10, 120),
            '연도': np.random.choice([2019, 2020, 2021, 2022, 2023], 120)
        }
        return pd.DataFrame(data)

# K-means 클러스터링 함수
def perform_kmeans(df, n_clusters=3):
    # 수치형 데이터만 선택
    numeric_cols = ['사고건수', '사망자수', '중상자수', '경상자수']
    X = df[numeric_cols].fillna(0)
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['클러스터'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans

# 메인 앱
def main():
    # 헤더
    st.markdown("# 🚸 경북 어린이 교통사고 분석 대시보드")
    st.markdown("---")
    
    # 데이터 로드
    df = load_data()
    
    # 사이드바
    with st.sidebar:
        st.markdown("## 📊 필터 설정")
        
        # 연도 선택
        years = sorted(df['연도'].unique()) if '연도' in df.columns else [2023]
        selected_year = st.selectbox("📅 연도 선택", years, index=len(years)-1)
        
        # 시군구 선택
        regions = ['전체'] + sorted(df['시군구'].unique().tolist())
        selected_region = st.selectbox("📍 지역 선택", regions)
        
        # 클러스터 개수
        n_clusters = st.slider("🎯 클러스터 개수", 2, 5, 3)
        
        st.markdown("---")
        st.markdown("### 📌 분석 정보")
        st.info("K-means 클러스터링을 통해 사고 유형을 분류합니다.")
    
    # 데이터 필터링
    filtered_df = df[df['연도'] == selected_year].copy() if '연도' in df.columns else df.copy()
    if selected_region != '전체':
        filtered_df = filtered_df[filtered_df['시군구'] == selected_region]
    
    # 상단 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_accidents = filtered_df['사고건수'].sum()
        st.metric("🚗 총 사고건수", f"{total_accidents:,}건")
    
    with col2:
        total_deaths = filtered_df['사망자수'].sum()
        st.metric("💀 사망자수", f"{total_deaths}명")
    
    with col3:
        total_serious = filtered_df['중상자수'].sum()
        st.metric("🏥 중상자수", f"{total_serious}명")
    
    with col4:
        total_minor = filtered_df['경상자수'].sum()
        st.metric("🩹 경상자수", f"{total_minor}명")
    
    st.markdown("---")
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📈 지역별 현황", "🎯 클러스터 분석", "📊 상세 통계"])
    
    with tab1:
        st.markdown("## 📍 지역별 사고 현황")
        
        # 지역별 집계
        region_stats = filtered_df.groupby('시군구').agg({
            '사고건수': 'sum',
            '사망자수': 'sum',
            '중상자수': 'sum',
            '경상자수': 'sum'
        }).reset_index()
        region_stats = region_stats.sort_values('사고건수', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 막대 그래프
            fig = px.bar(
                region_stats.head(10),
                x='시군구',
                y='사고건수',
                title='지역별 사고 건수 TOP 10',
                color='사고건수',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 파이 차트
            fig = px.pie(
                region_stats.head(5),
                values='사고건수',
                names='시군구',
                title='상위 5개 지역 사고 비율'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 순위 테이블
        st.markdown("### 🏆 지역별 순위")
        region_stats['순위'] = range(1, len(region_stats) + 1)
        st.dataframe(
            region_stats[['순위', '시군구', '사고건수', '사망자수', '중상자수', '경상자수']],
            use_container_width=True,
            hide_index=True
        )
    
    with tab2:
        st.markdown("## 🎯 K-Means 클러스터 분석")
        
        # 클러스터링 수행
        clustered_df, kmeans = perform_kmeans(filtered_df, n_clusters)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 산점도 - 사고건수 vs 사망자수
            fig = px.scatter(
                clustered_df,
                x='사고건수',
                y='사망자수',
                color='클러스터',
                size='중상자수',
                hover_data=['시군구'],
                title='클러스터별 사고 분포 (사고건수 vs 사망자수)',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 산점도 - 중상자수 vs 경상자수
            fig = px.scatter(
                clustered_df,
                x='중상자수',
                y='경상자수',
                color='클러스터',
                size='사고건수',
                hover_data=['시군구'],
                title='클러스터별 사고 분포 (중상자수 vs 경상자수)',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 클러스터별 통계
        st.markdown("### 📊 클러스터별 특성")
        cluster_stats = clustered_df.groupby('클러스터').agg({
            '사고건수': ['mean', 'sum'],
            '사망자수': ['mean', 'sum'],
            '중상자수': ['mean', 'sum'],
            '경상자수': ['mean', 'sum']
        }).round(2)
        
        st.dataframe(cluster_stats, use_container_width=True)
    
    with tab3:
        st.markdown("## 📊 상세 통계")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 시계열 분석 (연도별)
            if '연도' in df.columns:
                yearly_stats = df.groupby('연도')['사고건수'].sum().reset_index()
                fig = px.line(
                    yearly_stats,
                    x='연도',
                    y='사고건수',
                    title='연도별 사고 추이',
                    markers=True
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 피해 정도 비교
            damage_data = pd.DataFrame({
                '피해정도': ['사망', '중상', '경상'],
                '인원': [
                    filtered_df['사망자수'].sum(),
                    filtered_df['중상자수'].sum(),
                    filtered_df['경상자수'].sum()
                ]
            })
            fig = px.bar(
                damage_data,
                x='피해정도',
                y='인원',
                title='피해 정도별 인원',
                color='피해정도',
                color_discrete_map={'사망': '#ff4444', '중상': '#ff8800', '경상': '#ffcc00'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 원본 데이터
        with st.expander("🔍 원본 데이터 보기"):
            st.dataframe(filtered_df, use_container_width=True)
    
    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 20px;'>
            <p style='color: rgba(255,255,255,0.7);'>
                📊 2019-2023 경북 어린이 교통사고 데이터 분석<br>
                🏫 포항여자고등학교 | 최형배 선생님 지도
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
