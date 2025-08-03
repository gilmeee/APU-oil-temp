

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide", page_title="APU Oil Temp 예측 인터랙티브 대시보드")
st.title('APU Oil Temp 예측 인터랙티브 대시보드')

# --- 데이터 로딩 및 전처리 (새로운 머신러닝 로직에 맞춰 수정) ---
@st.cache_data
def load_and_engineer_features():
    """데이터 로드, 전처리, MALFUNCTION_ATA 피처 엔지니어링을 수행합니다."""
    try:
        df_full_data = pd.read_csv('APU_2023-06-29_2025-07-30_processedver2.csv')
        maint = pd.read_csv('APU_maint.csv')
        too_high = pd.read_csv('APU_high_temp_maint.csv')
        
    except FileNotFoundError as e:
        st.error(f"오류: '{e.filename}' 파일을 찾을 수 없습니다. CSV 파일들이 스크립트와 같은 폴더에 있는지 확인해주세요.")
        return None, None, None, None

    # --- 데이터 전처리 ---
    if 'Date' not in df_full_data.columns:
        st.error("오류: 'Date' 컬럼이 'APU_202306_202507_processed.csv' 파일에 없습니다.")
        return None, None, None, None
        
    df_full_data['Date'] = pd.to_datetime(df_full_data['Date'])
    df_full_data.rename(columns={'REGNO': 'HL_no'}, inplace=True)
    df_full_data = df_full_data.sort_values(by=['HL_no', 'Date'])
    df_full_data['month'] = df_full_data['Date'].dt.month
    df_full_data['month_sin'] = np.sin(2 * np.pi * df_full_data['month'] / 12)
    df_full_data['month_cos'] = np.cos(2 * np.pi * df_full_data['month'] / 12)
    
    maint['NR_REQUEST_DATE'] = pd.to_datetime(maint['NR_REQUEST_DATE'], errors='coerce')
    too_high['NR_REQUEST_DATE'] = pd.to_datetime(too_high['NR_REQUEST_DATE'], errors='coerce')

    # --- MALFUNCTION_ATA 피처 생성 
    # maint_for_merge = maint[['AC_NO', 'NR_REQUEST_DATE', 'MALFUNCTION_ATA']].copy()
    # maint_for_merge.rename(columns={'AC_NO': 'HL_no', 'NR_REQUEST_DATE': 'Date'}, inplace=True)
    # maint_for_merge.dropna(subset=['MALFUNCTION_ATA'], inplace=True)
    # maint_for_merge['MALFUNCTION_ATA'] = maint_for_merge['MALFUNCTION_ATA'].astype(str)
    
    # maint_ata_pivot = maint_for_merge.pivot_table(
    #     index=['HL_no', 'Date'],
    #     aggfunc=lambda x: 1,
    #     fill_value=0
    # ).reset_index()
    
    # maint_ata_pivot.columns = ['HL_no', 'Date'] + [f'ATA_{col}' for col in maint_ata_pivot.columns[2:]]
    
    # --- 데이터 병합 ---
    # df_full_data = pd.merge(df_full_data, maint_ata_pivot, on=['HL_no', 'Date'], how='left')
    new_ata_cols = [col for col in df_full_data.columns if col.startswith('ATA_')]
    df_full_data[new_ata_cols] = df_full_data[new_ata_cols].fillna(0)
    
    return df_full_data, maint, too_high, new_ata_cols

# --- 데이터 로드 실행 ---
df_full_data, maint, too_high, new_ata_cols = load_and_engineer_features()

# 데이터 로딩 성공 시에만 전체 앱 실행
if df_full_data is not None:
    # --- 사이드바 UI 구성 (스타일 및 구조는 그대로 유지) ---
    st.sidebar.header('⚙️ 모델 파라미터 선택')

    # CSS 스타일링
    st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #000080; color: white; border: none; }
    div.stButton > button:hover { background-color: #4682B4; color: white; border: none; }
    span[data-baseweb="tag"] { background-color: #000080 !important; }
    li[data-baseweb="menu-item-wrapper"]:hover { background-color: #4682B4; }
    </style>""", unsafe_allow_html=True)

    # --- UI 위젯 (드롭다운 방식 유지) ---
    base_numerical_features = ['TAT', 'EGTA_3', 'GLA_3', 'WB_3', 'PT_3', 'P2A_3', 'LCOT_3', 'LCIT_3', 'IGV_3', 'SCV_3', 'HOT_3', 'LOT_3']
    
    # 데이터에 실제 존재하는 컬럼만 선택지로 제공
    available_cols = [col for col in base_numerical_features if col in df_full_data.columns]
    all_features = ['N3EGTA', 'N3GLA', 'N3WB', 'N3PT', 'N3P2A', 'N3LCOT', 'N3LCIT', 'N3IGV', 'N3SCV', 'N3HOT', 'N3LOT']
    all_targets =  ['N3EGTA', 'N3GLA', 'N3WB', 'N3PT', 'N3P2A', 'N3LCOT', 'N3LCIT', 'N3IGV', 'N3SCV', 'N3HOT', 'N3LOT']
    available_tails = sorted(df_full_data['HL_no'].unique().astype(str))

    tail_nums = sorted(df_full_data['HL_no'].dropna().unique().tolist(), key=lambda x: int(''.join(filter(str.isdigit, x))))
    default_tail_index = tail_nums.index('HL8001') if 'HL8001' in tail_nums else 0
    selected_tail = st.sidebar.selectbox('1. 항공기 기번 선택:', available_tails, index=available_tails.index('HL8001'))
    selected_target = st.sidebar.selectbox('2. 예측 타겟 변수 선택:', all_targets)
    
    # 타겟으로 선택된 변수는 피처 목록에서 자동 제외
    available_features = [f for f in all_features if f != selected_target]
    selected_features = st.sidebar.multiselect('3. 학습 피처 변수 선택:', available_features, default=available_features)

    # --- 분석 시작 버튼 ---
    if st.sidebar.button('📊 통합 모델 학습 및 분석 시작'):
        with st.spinner('통합 모델 학습 및 결과 생성 중...'):
            
            # --- ✨✨✨ 새로운 머신러닝 로직 적용 ✨✨✨ ---
            target = selected_target
            # 사용자가 선택한 피처에 시간 관련 피처 추가
            numerical_features = selected_features + ['month_sin', 'month_cos']
            categorical_features = ['HL_no'] + new_ata_cols
            all_model_features = numerical_features + categorical_features

            train_start_date = pd.to_datetime('2023-06-01')
            train_end_date = pd.to_datetime('2024-05-31')
            
            df_train_period = df_full_data[(df_full_data['Date'] >= train_start_date) & (df_full_data['Date'] <= train_end_date)].copy()
            
            if target not in df_train_period.columns:
                 st.error(f"타겟 변수 '{target}'가 데이터에 없습니다. 데이터나 코드의 타겟 변수명을 확인해주세요.")
            else:
                df_train_period.dropna(subset=[target], inplace=True)
                
                X_train_overall = df_train_period[all_model_features]
                y_train_overall = df_train_period[target]

                # --- 모델 파이프라인 정의 ---
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

                # --- 최종 통합 모델 학습 ---
                pipeline.fit(X_train_overall, y_train_overall)
                st.success("✅ 통합 모델 학습이 완료되었습니다.")

                # --- 선택된 항공기에 대한 예측 및 시각화 ---
                st.markdown("---")
                st.header(f"항공기 {selected_tail} 분석 결과")

                df_selected_tail = df_full_data[df_full_data['HL_no'] == selected_tail].copy()
                
                if target not in df_selected_tail.columns:
                    st.error(f"타겟 변수 '{target}'가 선택된 항공기 데이터에 없습니다.")
                else:
                    df_selected_tail.dropna(subset=[target], inplace=True)

                    if df_selected_tail.empty:
                        st.warning(f"항공기 {selected_tail}에 대한 유효한 데이터가 부족하여 결과를 표시할 수 없습니다.")
                    else:
                        # 예측 수행
                        predictions = pipeline.predict(df_selected_tail[all_model_features])
                        df_selected_tail['Predicted'] = predictions
                        df_selected_tail['Residual'] = df_selected_tail[target] - df_selected_tail['Predicted']

                        # 시각화
                        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
                        
                        # 상단 그래프: 실제값 vs 예측값
                        sns.lineplot(x='Date', y=target, data=df_selected_tail, label=f'Actual {target}', color='blue', ax=axes[0], marker='o', markersize=3, alpha=0.7)
                        sns.lineplot(x='Date', y='Predicted', data=df_selected_tail, label=f'Predicted {target}', color='red', linestyle='--', ax=axes[0])
                        axes[0].axvline(x=train_end_date, color='purple', linestyle=':', linewidth=2, label=f'Training End ({train_end_date.date()})')
                        
                        # 정비 및 고장 이력 시각화
                        fault_dates = too_high[too_high['AC_NO'] == selected_tail]['NR_REQUEST_DATE'].dropna()
                        maint_dates = maint[maint['AC_NO'] == selected_tail]['NR_REQUEST_DATE'].dropna()
                        
                        # 범례에 한 번만 표시하기 위한 플래그
                        plotted_labels = {'fault': False, 'maint': False}
                        for date in fault_dates:
                            label = 'High Temp Fault' if not plotted_labels['fault'] else ""
                            axes[0].axvline(x=date, color='black', linestyle='-', linewidth=1, alpha=0.8, label=label)
                            plotted_labels['fault'] = True
                        for date in maint_dates:
                            label = 'Maint. Record' if not plotted_labels['maint'] else ""
                            axes[0].axvline(x=date, color='orange', linestyle='--', linewidth=1, alpha=0.8, label=label)
                            plotted_labels['maint'] = True

                        axes[0].set_title(f'Actual vs. Predicted for {target} (Aircraft: {selected_tail})')
                        axes[0].set_ylabel(f'{target} Value')
                        axes[0].legend()
                        axes[0].grid(True)

                        # 하단 그래프: 잔차
                        sns.lineplot(x='Date', y='Residual', data=df_selected_tail, color='green', ax=axes[1])
                        axes[1].axhline(y=0, color='gray', linestyle='--')
                        axes[1].set_title('Residuals over Time')
                        axes[1].set_xlabel('Date')
                        axes[1].set_ylabel('Residual Value')
                        axes[1].grid(True)
                        
                        plt.tight_layout()
                        st.pyplot(fig)

                        # 정비 기록 테이블
                        st.write("#### 전체 기간 정비 기록")
                        maint_records = maint[maint['AC_NO'] == selected_tail].copy()
                        if not maint_records.empty:
                            st.dataframe(maint_records[['NR_REQUEST_DATE', 'CORRECTIVE_ACTION']].sort_values(by='NR_REQUEST_DATE', ascending=False))
                        else:
                            st.info("해당 항공기에 대한 정비 기록이 없습니다.")
