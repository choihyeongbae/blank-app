import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import koreanize_matplotlib

# 페이지 설정
st.set_page_config(
    page_title="어린이 교통사고 지역 분석",
    page_icon="🚦",
    layout="wide"
)

# 타이틀
st.title("🚦 어린이 교통사고 지역별 안전도 분석")
st.markdown("---")

# 데이터 로드 함수
@st.cache_data
def load_and_cluster_data():
    """데이터를 로드하고 K-Means 클러스터링을 수행합니다."""
    # CSV 파일 로드 (사용자가 업로드하거나 경로 지정)
    try:
        df = pd.read_csv("Rates_by_Age_2024.csv")
    except FileNotFoundError:
        st.error("⚠️ 'Rates_by_Age_2024.csv' 파일을 찾을 수 없습니다. 파일을 업로드해주세요.")
        return None, None, None, None
    
    # K-Means 클러스터링
    X = pd.DataFrame({
        '부상자수': df['Injuries_1000'].values,
        '사고건수': df['Accidents_1000'].values,
    })
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = cluster_labels
    
    # 실루엣 스코어 계산
    sil_score = silhouette_score(X_scaled, cluster_labels)
    
    # 중심점 계산
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return df, sil_score, centroids_original, scaler

# 파일 업로드 옵션
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (선택사항)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # K-Means 클러스터링
    X = pd.DataFrame({
        '부상자수': df['Injuries_1000'].values,
        '사고건수': df['Accidents_1000'].values,
    })
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = cluster_labels
    
    sil_score = silhouette_score(X_scaled, cluster_labels)
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
else:
    df, sil_score, centroids_original, scaler = load_and_cluster_data()

if df is not None:
    # 사이드바: 검색 기능
    st.sidebar.header("🔍 지역 검색")
    
    # 지역 목록
    locations = sorted(df['Location'].unique())
    selected_location = st.sidebar.selectbox(
        "지역을 선택하세요:",
        options=["전체"] + locations
    )
    
    # 클러스터 정보 정의
    cluster_info = {
        0: {"name": "안전 지역", "color": "🟢", "description": "비교적 안전한 지역"},
        1: {"name": "주의 지역", "color": "🟡", "description": "중간 위험 지역"},
        2: {"name": "위험 지역", "color": "🔴", "description": "고위험 지역"}
    }
    
    # 메인 영역
    if selected_location != "전체":
        # 선택된 지역 정보
        region_data = df[df['Location'] == selected_location].iloc[0]
        cluster_num = region_data['cluster']
        
        # 지역 정보 표시
        st.header(f"📍 {selected_location}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="클러스터 분류",
                value=f"{cluster_info[cluster_num]['color']} {cluster_info[cluster_num]['name']}"
            )
        
        with col2:
            st.metric(
                label="1000명당 부상자수",
                value=f"{region_data['Injuries_1000']:.2f}명"
            )
        
        with col3:
            st.metric(
                label="1000명당 사고건수",
                value=f"{region_data['Accidents_1000']:.2f}건"
            )
        
        with col4:
            # 전국 평균 대비
            avg_injuries = df['Injuries_1000'].mean()
            diff = region_data['Injuries_1000'] - avg_injuries
            st.metric(
                label="전국 평균 대비",
                value=f"{diff:+.2f}명",
                delta=f"{(diff/avg_injuries)*100:.1f}%"
            )
        
        st.markdown("---")
        
        # 시각화 섹션
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📊 전국 클러스터 분포")
            
            # 클러스터링 결과 산점도
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = {0: 'orange', 1: 'blue', 2: 'green'}
            labels = {0: '군집 0 (안전)', 1: '군집 1 (주의)', 2: '군집 2 (위험)'}
            
            for c in df['cluster'].unique():
                temp = df[df['cluster'] == c]
                ax.scatter(
                    temp['Injuries_1000'],
                    temp['Accidents_1000'],
                    label=labels[c],
                    s=100,
                    c=colors[c],
                    alpha=0.6,
                    zorder=8
                )
            
            # 선택된 지역 강조
            ax.scatter(
                region_data['Injuries_1000'],
                region_data['Accidents_1000'],
                s=500,
                c='red',
                marker='*',
                edgecolors='black',
                linewidths=2,
                label=f'{selected_location} (선택)',
                zorder=10
            )
            
            # 중심점
            ax.scatter(
                centroids_original[:, 0],
                centroids_original[:, 1],
                marker='X',
                s=300,
                color='black',
                label='중심점',
                zorder=9
            )
            
            ax.set_xlabel('Injuries per 1000 children', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accidents per 1000 children', fontsize=12, fontweight='bold')
            ax.set_title('K-Means Clustering Result (K=3)', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            plt.close()
        
        with col_right:
            st.subheader("📈 지역별 비교")
            
            # 같은 클러스터 내 지역들과 비교
            same_cluster = df[df['cluster'] == cluster_num].sort_values('Injuries_1000', ascending=False)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # 부상자수 비교
            top_10 = same_cluster.head(10)
            colors_bar = ['red' if loc == selected_location else 'skyblue' for loc in top_10['Location']]
            
            ax1.barh(range(len(top_10)), top_10['Injuries_1000'], color=colors_bar)
            ax1.set_yticks(range(len(top_10)))
            ax1.set_yticklabels(top_10['Location'])
            ax1.set_xlabel('1000명당 부상자수', fontweight='bold')
            ax1.set_title(f'{cluster_info[cluster_num]["name"]} 내 부상자수 상위 10개 지역', fontweight='bold')
            ax1.axvline(region_data['Injuries_1000'], color='red', linestyle='--', linewidth=2)
            ax1.grid(axis='x', linestyle='--', alpha=0.6)
            
            # 사고건수 비교
            same_cluster_acc = same_cluster.sort_values('Accidents_1000', ascending=False).head(10)
            colors_bar2 = ['red' if loc == selected_location else 'lightcoral' for loc in same_cluster_acc['Location']]
            
            ax2.barh(range(len(same_cluster_acc)), same_cluster_acc['Accidents_1000'], color=colors_bar2)
            ax2.set_yticks(range(len(same_cluster_acc)))
            ax2.set_yticklabels(same_cluster_acc['Location'])
            ax2.set_xlabel('1000명당 사고건수', fontweight='bold')
            ax2.set_title(f'{cluster_info[cluster_num]["name"]} 내 사고건수 상위 10개 지역', fontweight='bold')
            ax2.axvline(region_data['Accidents_1000'], color='red', linestyle='--', linewidth=2)
            ax2.grid(axis='x', linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 상세 통계
        st.markdown("---")
        st.subheader("📋 상세 통계")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**전국 평균**")
            st.write(f"부상자수: {df['Injuries_1000'].mean():.2f}명")
            st.write(f"사고건수: {df['Accidents_1000'].mean():.2f}건")
        
        with col2:
            st.write(f"**{cluster_info[cluster_num]['name']} 평균**")
            cluster_data = df[df['cluster'] == cluster_num]
            st.write(f"부상자수: {cluster_data['Injuries_1000'].mean():.2f}명")
            st.write(f"사고건수: {cluster_data['Accidents_1000'].mean():.2f}건")
        
        with col3:
            st.write(f"**{selected_location} 순위**")
            rank_injuries = (df['Injuries_1000'] > region_data['Injuries_1000']).sum() + 1
            rank_accidents = (df['Accidents_1000'] > region_data['Accidents_1000']).sum() + 1
            st.write(f"부상자수: {rank_injuries}/{len(df)}위")
            st.write(f"사고건수: {rank_accidents}/{len(df)}위")
    
    else:
        # 전체 개요
        st.header("📊 전국 어린이 교통사고 안전도 개요")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="전체 분석 지역",
                value=f"{len(df)}개"
            )
        
        with col2:
            st.metric(
                label="안전 지역 (군집 0)",
                value=f"{(df['cluster']==0).sum()}개"
            )
        
        with col3:
            st.metric(
                label="주의 지역 (군집 1)",
                value=f"{(df['cluster']==1).sum()}개"
            )
        
        with col4:
            st.metric(
                label="위험 지역 (군집 2)",
                value=f"{(df['cluster']==2).sum()}개"
            )
        
        st.markdown("---")
        
        # 전체 시각화
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🗺️ 전국 클러스터 분포")
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            colors = {0: 'orange', 1: 'blue', 2: 'green'}
            labels = {0: '군집 0 (안전)', 1: '군집 1 (주의)', 2: '군집 2 (위험)'}
            
            for c in df['cluster'].unique():
                temp = df[df['cluster'] == c]
                ax.scatter(
                    temp['Injuries_1000'],
                    temp['Accidents_1000'],
                    label=labels[c],
                    s=100,
                    c=colors[c],
                    alpha=0.6,
                    zorder=8
                )
            
            ax.scatter(
                centroids_original[:, 0],
                centroids_original[:, 1],
                marker='X',
                s=300,
                color='black',
                label='중심점',
                zorder=9
            )
            
            ax.set_xlabel('Injuries per 1000 children', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accidents per 1000 children', fontsize=12, fontweight='bold')
            ax.set_title('K-Means Clustering Result (K=3)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            plt.close()
        
        with col_right:
            st.subheader("📊 클러스터별 통계")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # 클러스터별 평균 부상자수
            cluster_means_inj = df.groupby('cluster')['Injuries_1000'].mean()
            colors_cluster = ['orange', 'blue', 'green']
            
            ax1.bar(cluster_means_inj.index, cluster_means_inj.values, color=colors_cluster)
            ax1.set_xlabel('클러스터', fontweight='bold')
            ax1.set_ylabel('평균 부상자수', fontweight='bold')
            ax1.set_title('클러스터별 평균 부상자수 (1000명당)', fontweight='bold')
            ax1.set_xticks([0, 1, 2])
            ax1.set_xticklabels(['군집 0\n(안전)', '군집 1\n(주의)', '군집 2\n(위험)'])
            ax1.grid(axis='y', linestyle='--', alpha=0.6)
            
            # 클러스터별 평균 사고건수
            cluster_means_acc = df.groupby('cluster')['Accidents_1000'].mean()
            
            ax2.bar(cluster_means_acc.index, cluster_means_acc.values, color=colors_cluster)
            ax2.set_xlabel('클러스터', fontweight='bold')
            ax2.set_ylabel('평균 사고건수', fontweight='bold')
            ax2.set_title('클러스터별 평균 사고건수 (1000명당)', fontweight='bold')
            ax2.set_xticks([0, 1, 2])
            ax2.set_xticklabels(['군집 0\n(안전)', '군집 1\n(주의)', '군집 2\n(위험)'])
            ax2.grid(axis='y', linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 상위/하위 지역 표
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚠️ 부상자수 상위 10개 지역")
            top_injuries = df.nlargest(10, 'Injuries_1000')[['Location', 'Injuries_1000', 'Accidents_1000', 'cluster']]
            top_injuries['클러스터'] = top_injuries['cluster'].map(lambda x: cluster_info[x]['name'])
            st.dataframe(
                top_injuries[['Location', 'Injuries_1000', 'Accidents_1000', '클러스터']].rename(columns={
                    'Location': '지역',
                    'Injuries_1000': '부상자수',
                    'Accidents_1000': '사고건수'
                }),
                hide_index=True,
                use_container_width=True
            )
        
        with col2:
            st.subheader("✅ 부상자수 하위 10개 지역")
            bottom_injuries = df.nsmallest(10, 'Injuries_1000')[['Location', 'Injuries_1000', 'Accidents_1000', 'cluster']]
            bottom_injuries['클러스터'] = bottom_injuries['cluster'].map(lambda x: cluster_info[x]['name'])
            st.dataframe(
                bottom_injuries[['Location', 'Injuries_1000', 'Accidents_1000', '클러스터']].rename(columns={
                    'Location': '지역',
                    'Injuries_1000': '부상자수',
                    'Accidents_1000': '사고건수'
                }),
                hide_index=True,
                use_container_width=True
            )
    
    # 푸터
    st.markdown("---")
    st.info(f"""
    **📌 분석 정보**
    - 클러스터링 알고리즘: K-Means (K=3)
    - 실루엣 스코어: {sil_score:.3f}
    - 분석 지역 수: {len(df)}개
    - 데이터 기준: 2024년 어린이 인구 1000명당 교통사고 지표
    """)

else:
    st.warning("⚠️ CSV 파일을 업로드하거나 'Rates_by_Age_2024.csv' 파일을 확인해주세요.")
