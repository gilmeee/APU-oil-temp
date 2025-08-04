import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- 기본 설정 ---
# 경고 메시지 무시
warnings.filterwarnings('ignore')
# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="APU 통합 모델 예측 대시보드")
# Matplotlib 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- CSS 스타일 ---
st.markdown("""
<style>
    /* 분석 시작 버튼 스타일 */
    div.stButton > button:first-child {
        background-color: #000080; /* 남색 */
        color: white;
        border: none;
        border-radius: 8px; /* 모서리 둥글게 */
        padding: 16px 32px; /* 버튼 크기 (상하, 좌우 여백) */
        font-size: 18px; /* 글자 크기 */
        font-weight: bold; /* 글자 굵게 */
        width: 100%; /* 사이드바 너비에 꽉 차게 */
    }
    div.stButton > button:hover {
        background-color: #4682B4; /* 밝은 남색 */
        color: white;
        border: none;
    }
    /* 멀티셀렉트 선택된 항목 스타일 */
    span[data-baseweb="tag"] {
        background-color: #000080 !important;
    }
    
    /* 드롭다운(Selectbox) 및 멀티셀렉트(MultiSelect) 기본 스타일 */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] {
        background-color: white;
        border: 1px solid #000080; /* 테두리: 1px 두께의 남색 실선 */
        border-radius: 5px;
        box-shadow: none; /* 기본 그림자 제거 */
    }

    /* 포커스(클릭) 시 스타일: Streamlit 기본 빨간색 테두리 덮어쓰기 */
    div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"]:focus-within {
         border-color: #000080; /* 클릭 시 남색 테두리 유지 */
         box-shadow: 0 0 0 2px #4682B4 !important; /* !important로 기본 스타일 강제 덮어쓰기 */
         outline: none;
    }
</style>""", unsafe_allow_html=True)

# --- 타이틀 ---
st.title('APU 통합 예측 모델 대시보드')
st.caption('모든 항공기 데이터를 학습한 단일 통합 모델을 사용하여 특정 항공기의 상태를 예측하고 분석합니다.')

# --- 데이터 로딩 및 전처리 (캐싱으로 속도 극대화) ---
@st.cache_data
def preprocess_data():
    """
    데이터를 로드하고 시간 특성, 정비 특성(MALFUNCTION_ATA)을 생성 및 병합합니다.
    이 함수는 앱 실행 시 단 한 번만 실행됩니다.
    """
    try:
        # 1. 데이터 로드
        df_data = pd.read_csv('APU_2023-06-29_2025-07-30_processedver2.csv')
        too_high = pd.read_csv('APU_high_temp_maint.csv')
        maint = pd.read_csv('APU_maint.csv')

        # 2. 기본 전처리 (날짜 변환 및 이름 통일)
        df_data.rename(columns={'CREATION_DATE': 'DATE', 'REGNO': 'AC_NO'}, inplace=True)
        df_data['DATE'] = pd.to_datetime(df_data['DATE'])
        too_high['NR_REQUEST_DATE'] = pd.to_datetime(too_high['NR_REQUEST_DATE'], errors='coerce')
        maint['NR_REQUEST_DATE'] = pd.to_datetime(maint['NR_REQUEST_DATE'], errors='coerce')
        
        # 3. 시간 관련 특성 생성
        df_data = df_data.sort_values(by=['AC_NO', 'DATE'])
        df_data['hour'] = df_data['DATE'].dt.hour
        df_data['month'] = df_data['DATE'].dt.month
        df_data['dayofweek'] = df_data['DATE'].dt.dayofweek
        df_data['month_sin'] = np.sin(2 * np.pi * df_data['month'] / 12)
        df_data['month_cos'] = np.cos(2 * np.pi * df_data['month'] / 12)

        # 4. MALFUNCTION 특성 생성 (MALFUNCTION_ATA X)
        maint_for_merge = maint[['AC_NO', 'NR_REQUEST_DATE', 'MALFUNCTION_ATA']].copy()
        maint_for_merge.rename(columns={'NR_REQUEST_DATE': 'DATE'}, inplace=True)
        maint_for_merge.dropna(subset=['MALFUNCTION_ATA'], inplace=True)
        maint_for_merge['MALFUNCTION_ATA'] = maint_for_merge['MALFUNCTION_ATA'].astype(str)

        maint_ata_pivot = maint_for_merge.pivot_table(
            index=['AC_NO', 'DATE'],
            columns='MALFUNCTION_ATA',
            aggfunc=lambda x: 1,
            fill_value=0
        ).reset_index()
        
        # 멀티 인덱스 컬럼 이름 정리
        new_cols = [f'ATA_{col}' for col in maint_ata_pivot.columns if col not in ['AC_NO', 'DATE']]
        maint_ata_pivot.columns = ['AC_NO', 'DATE'] + new_cols
        
        # 5. 원본 데이터에 ATA 특성 병합
        df_processed = pd.merge(df_data, maint_ata_pivot, on=['AC_NO', 'DATE'], how='left')
        ata_cols = [col for col in df_processed.columns if col.startswith('ATA_')]
        df_processed[ata_cols] = df_processed[ata_cols].fillna(0)

        return df_processed, too_high, maint

    except FileNotFoundError as e:
        st.error(f"오류: '{e.filename}' 파일을 찾을 수 없습니다. CSV 파일들이 스크립트와 같은 폴더에 있는지 확인해주세요.")
        return None, None, None

# --- 통합 모델 학습 (캐싱으로 동일 조건 재학습 방지) ---
@st.cache_data
def train_unified_model(df, target, numerical_features, categorical_features):
    """
    전체 데이터를 받아 통합 모델을 학습하고, 시계열 교차검증을 수행합니다.
    target, features가 바뀔 때만 재학습됩니다.
    """
    # 1. 학습/테스트 기간 정의
    train_start = pd.to_datetime("2023-06-01")
    train_end = pd.to_datetime("2024-05-31")
    df_train = df[(df['DATE'] >= train_start) & (df['DATE'] <= train_end)].copy()

    X_train = df_train[numerical_features + categorical_features]
    y_train = df_train[target]

    # 2. 모델 파이프라인 구축
    preprocessor = ColumnTransformer(
        [
            ('scaler_numerical', StandardScaler(), numerical_features),
            ('onehot_categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso(alpha=0.01, max_iter=10000))
    ])

    # 3. 시계열 교차 검증 (TimeSeriesSplit)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = {'MAE': [], 'RMSE': [], 'R2': []}

    for train_index, test_index in tscv.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        pipeline.fit(X_train_fold, y_train_fold)
        preds = pipeline.predict(X_test_fold)
        
        cv_scores['MAE'].append(mean_absolute_error(y_test_fold, preds))
        cv_scores['RMSE'].append(np.sqrt(mean_squared_error(y_test_fold, preds)))
        cv_scores['R2'].append(r2_score(y_test_fold, preds))

    # 4. 최종 모델 학습 (전체 학습 데이터 사용)
    pipeline.fit(X_train, y_train)

    return pipeline, cv_scores


# --- 데이터 로드 ---
df_processed, too_high, maint = preprocess_data()

# 데이터 로딩 성공 시에만 전체 앱 실행
if df_processed is not None:
    # --- 사이드바 UI 구성 ---
    st.sidebar.header('⚙️ 모델 파라미터 선택')

    # 사용 가능한 피처 및 타겟 목록 정의
    base_features = ['N3EGTA', 'N3GLA', 'N3WB', 'N3PT', 'N3P2A', 'N3LCOT', 'N3LCIT', 'N3IGV', 'N3SCV', 'N3HOT', 'N3LOT']
    time_features = ['hour', 'month_sin', 'month_cos', 'dayofweek']

    # UI 표시용 이름 생성 함수
    format_feature_name = lambda name: name.replace('N3', '')

    
    # 1. 분석할 항공기 선택
    available_tails = sorted(df_processed['AC_NO'].unique().astype(str))
    selected_tail = st.sidebar.selectbox('1. 분석할 항공기 선택:', available_tails, index=available_tails.index('HL8001'))
    
    # 2. 예측 타겟 변수 선택
    selected_target = st.sidebar.selectbox(
        '2. 예측 타겟 변수 선택:',
        options=base_features,
        index=len(base_features)-1, # N3LOT를 기본값으로
        format_func=format_feature_name # UI에 표시될 이름 포맷 지정
    )
    
    # 3. 학습 피처 변수 선택 (타겟으로 선택된 변수는 피처에서 제외)
    available_features_for_ui = [f for f in base_features if f != selected_target]
    selected_base_features = st.sidebar.multiselect(
        '3. 학습 피처 선택:',
        options=available_features_for_ui,
        default=available_features_for_ui,
        format_func=format_feature_name # UI에 표시될 이름 포맷 지정
    )

    # --- 분석 시작 버튼 ---
    if st.sidebar.button('📊 분석 시작', type="primary"):
        with st.spinner('통합 모델 학습 및 결과 분석 중...'):
            
            # # --- 1. 모델 학습 및 교차검증 ---
            # st.subheader("1. 통합 모델 학습 및 검증")
            
            # 모델에 사용될 최종 피처 목록 정의
            selected_features = selected_base_features + time_features
            numerical_features = [f for f in selected_features if f in base_features + time_features]
            ata_features = [col for col in df_processed.columns if col.startswith('ATA_')]
            categorical_features = ['AC_NO'] + ata_features
            all_model_features = numerical_features + categorical_features

            # 모델 학습 함수 호출
            trained_pipeline, cv_scores = train_unified_model(df_processed, selected_target, numerical_features, categorical_features)
            
            # 표시용 타겟 이름 생성
            display_target_name = format_feature_name(selected_target)

            # # 교차검증 결과 표시
            # st.write("시계열 교차검증(TimeSeriesSplit, n=5) 평균 성능:")
            # col1, col2, col3 = st.columns(3)
            # col1.metric("MAE (Mean Absolute Error)", f"{np.mean(cv_scores['MAE']):.3f}")
            # col2.metric("RMSE (Root Mean Squared Error)", f"{np.mean(cv_scores['RMSE']):.3f}")
            # col3.metric("R² Score", f"{np.mean(cv_scores['R2']):.3f}")
            
            # st.markdown("---")

            # --- 2. 선택한 항공기 예측 및 분석 ---
            st.subheader(f"{selected_tail} - {display_target_name} 예측 분석 (2025/01~2025/08)")

            # 전체 데이터에 대해 예측 수행
            df_processed['Predicted'] = trained_pipeline.predict(df_processed[all_model_features])
            df_processed['Residual'] = df_processed[selected_target] - df_processed['Predicted']

            # 사용자가 선택한 항공기 데이터만 필터링
            df_tail_analysis = df_processed[df_processed['AC_NO'] == selected_tail].copy()

            # 예측 기간(2025.01 ~ 2025.08) 데이터만 필터링하여 시각화
            start_date = pd.to_datetime('2025-01-01')
            end_date = pd.to_datetime('2025-08-31')
            df_plot = df_tail_analysis[(df_tail_analysis['DATE'] >= start_date) & (df_tail_analysis['DATE'] <= end_date)]

            if df_plot.empty:
                st.warning(f"{selected_tail}에 대한 예측 기간(2025/01~2025/08) 데이터가 없습니다.")
            else:
                # 시각화
                fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
                
                # 정비/결함 기록 필터링 (예측 기간 내 기록만)
                fault_dates = too_high[(too_high['AC_NO'] == selected_tail) & (too_high['NR_REQUEST_DATE'].between(start_date, end_date))]['NR_REQUEST_DATE'].dropna()
                maint_dates = maint[(maint['AC_NO'] == selected_tail) & (maint['NR_REQUEST_DATE'].between(start_date, end_date))]['NR_REQUEST_DATE'].dropna()
                
                # 상단 그래프: 실제값 vs 예측값
                sns.lineplot(x='DATE', y=selected_target, data=df_plot, label=f'실제 {display_target_name}', color='blue', ax=axes[0], marker='o', markersize=3, alpha=0.7)
                sns.lineplot(x='DATE', y='Predicted', data=df_plot, label=f'예측 {display_target_name}', color='red', linestyle='--', ax=axes[0])
                
                for i, date in enumerate(fault_dates):
                    axes[0].axvline(x=date, color='black', linestyle='-', linewidth=1.5, label='고온 결함' if i == 0 else "")
                
                for i, date in enumerate(maint_dates.unique()):
                    axes[0].axvline(x=date, color='darkorange', linestyle=':', linewidth=2, alpha=0.9, label='정비 기록' if i == 0 else "")

                axes[0].set_title(f'{selected_tail} - {display_target_name} 예측값 분석 (2025/01 ~ 2025/08)', fontsize=16)
                axes[0].set_ylabel(f'{display_target_name} 값')
                axes[0].grid(True, linestyle='--', alpha=0.6)
                axes[0].legend()

                # 하단 그래프: 잔차(Residual)
                axes[1].fill_between(df_plot['DATE'], df_plot['Residual'], 0, where=(df_plot['Residual'] > 0), color='red', alpha=0.3, label='과소 예측 (실제값 > 예측값)')
                axes[1].fill_between(df_plot['DATE'], df_plot['Residual'], 0, where=(df_plot['Residual'] < 0), color='blue', alpha=0.3, label='과대 예측 (실제값 < 예측값)')
                axes[1].axhline(y=0, color='gray', linestyle='--')
                
                # 잔차 그래프에도 정비/결함 라인 추가
                for i, date in enumerate(fault_dates):
                    axes[1].axvline(x=date, color='black', linestyle='-', linewidth=1.5, label='고온 결함' if i == 0 else "")
                
                for i, date in enumerate(maint_dates.unique()):
                    axes[1].axvline(x=date, color='darkorange', linestyle=':', linewidth=2, alpha=0.9, label='정비 기록' if i == 0 else "")

                axes[1].set_title(f'예측 잔차 (Residuals)', fontsize=16)
                axes[1].set_xlabel('날짜')
                axes[1].set_ylabel('잔차 값')
                axes[1].grid(True, linestyle='--', alpha=0.6)
                axes[1].legend()

                plt.tight_layout()
                st.pyplot(fig)

                # 정비 기록 테이블 출력 (예측 기간 내 기록만)
                st.write(f"{selected_tail} 정비 기록 (2025/01 ~ 2025/08)")
                maint_records = maint[(maint['AC_NO'] == selected_tail) & (maint['NR_REQUEST_DATE'].between(start_date, end_date))].copy()
                if not maint_records.empty:
                    maint_records['DATE_STR'] = maint_records['NR_REQUEST_DATE'].dt.strftime('%Y-%m-%d')
                    # 오래된 날짜 순으로 정렬하고, 인덱스를 1부터 시작하도록 수정
                    display_df = maint_records[['DATE_STR', 'MALFUNCTION_ATA', 'MALFUNCTION', 'CORRECTIVE_ACTION']].sort_values(by='DATE_STR', ascending=True)
                    display_df.index = np.arange(1, len(display_df) + 1)
                    st.dataframe(display_df)
                else:
                    st.info("해당 기간에 정비 기록이 없습니다.")