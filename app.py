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
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
import matplotlib.font_manager as fm

# --- 기본 설정 ---
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="APU 결함 예측 및 데이터 분석 대시보드")

# --- 폰트 파일 인코딩 함수 (CSS 전용) ---
def encode_font(font_path: str):
    if not os.path.exists(font_path):
        # CSS용 폰트가 없어도 앱은 돌아가게 하고, 경고만 표시
        st.warning(f"CSS용 폰트 파일을 찾을 수 없습니다: {font_path}")
        return None
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# --- Matplotlib 한글 폰트 설정 (그래프 전용) ---
def set_matplotlib_korean_font():
    # 1) 로컬에 둔 Hanjin 폰트가 있으면 최우선 사용
    for p in ["HanjinGroupSans.ttf", "HanjinGroupSansBold.ttf"]:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                name = fm.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = name
                break
            except Exception:
                pass
    else:
        # 2) 없으면 matplotlib 내장 폰트로 안전하게
        plt.rcParams["font.family"] = "DejaVu Sans"

    # 음수 부호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.family"]


# (선택) 디버그: 실제 적용된 폰트명 확인하고 싶으면 다음 줄 주석 해제
# st.caption(f"matplotlib font → {set_matplotlib_korean_font()}")
set_matplotlib_korean_font()
# --- 폰트 인코딩 실행 ---
# 일반 글씨체와 굵은 글씨체를 각각 불러옵니다.
regular_font_encoded = encode_font("HanjinGroupSans.ttf")
bold_font_encoded = encode_font("HanjinGroupSansBold.ttf")

# --- CSS 스타일 ---
# 폰트 인코딩이 성공했을 경우에만 CSS를 적용합니다.
if regular_font_encoded and bold_font_encoded:
    st.markdown(f"""
    <style>
        /* 1. 폰트 정의: 일반체와 굵은체를 모두 등록합니다 */
        @font-face {{
            font-family: 'Hanjin Group Sans';
            src: url(data:font/ttf;base64,{regular_font_encoded}) format('truetype');
            font-weight: normal;
        }}

        @font-face {{
            font-family: 'Hanjin Group Sans';
            src: url(data:font/ttf;base64,{bold_font_encoded}) format('truetype');
            font-weight: bold;
        }}

        /* 2. 폰트 적용: 제목(h1) 등을 포함한 모든 요소에 강제로 적용합니다 */
        html, body, [class*="st-"], h1, h2, h3, h4, h5, h6 {{
            font-family: 'Hanjin Group Sans', sans-serif !important;
        }}

        /* Material Icon class에는 저희 폰트가 적용되지 않도록 예외 처리 */
        .material-icons, .MuiIcon-root, .st-icon {{
            font-family: 'Material Icons' !important;
            font-style: normal;
            font-weight: normal;
            font-size: 24px;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            display: inline-block;
            white-space: nowrap;
            direction: ltr;
            -webkit-font-feature-settings: 'liga';
            -webkit-font-smoothing: antialiased;
        }}

        /* --- 이하 기존 스타일 유지 --- */
        .stDataFrame th, .stDataFrame td {{
            text-align: center !important;
        }}
        .stDataFrame thead tr th:first-child, .stDataFrame tbody tr th {{
            text-align: left !important;
        }}
        div.stButton > button:first-child {{
            background-color: #000080; color: white; border: none;
            border-radius: 8px; padding: 16px 32px; font-size: 18px;
            font-weight: bold; width: 100%;
        }}
        div.stButton > button:hover {{
            background-color: #4682B4; color: white; border: none;
        }}
        span[data-baseweb="tag"], [data-testid="stTag"] {{
            background-color: #000080 !important;
            color: white !important;
        }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"],
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] {{
            background-color: white; border: 1px solid #000080;
            border-radius: 5px; box-shadow: none;
        }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"]:focus-within {{
            border-color: #000080;
            box-shadow: 0 0 0 2px #4682B4 !important;
            outline: none;
        }}
        .legend-container {{
            display: flex; gap: 20px; margin-top: -10px; margin-bottom: 10px;
        }}
        .legend-item {{
            display: flex; align-items: center; font-size: 14px;
        }}
        .color-box {{
            width: 20px; height: 20px; margin-right: 8px; border: 1px solid #ccc;
        }}
    </style>
    """, unsafe_allow_html=True)

# --- Matplotlib 한글 폰트 설정 ---
# 그래프의 한글은 안정적인 '맑은 고딕'을 사용하도록 직접 지정합니다.
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
# ----------------------------------

# --- 타이틀 ---
st.title('APU 결함 예측 및 데이터 분석 대시보드')
st.caption('모든 항공기 데이터를 학습한 단일 통합 모델을 사용하여 특정 항공기의 상태를 예측하고 분석합니다.') #made with 한진 폰트

# --- 데이터 로딩 및 전처리 (캐싱으로 속도 극대화) ---
@st.cache_data
def preprocess_data():
    try:
        # 1. 데이터 로드
        df_data = pd.read_csv('APU_2023-06-29_2025-07-30_processedver2.csv')
        maint = pd.read_csv('APU_maint.csv')
        
        # 컬럼 이름에서 'N3' 제거
        df_data.rename(columns={
            'CREATION_DATE': 'DATE',
            'REGNO': 'AC_NO',
            'N3EGTA': 'EGTA',
            'N3GLA': 'GLA',
            'N3WB': 'WB',
            'N3PT': 'PT',
            'N3P2A': 'P2A',
            'N3LCOT': 'LCOT',
            'N3LCIT': 'LCIT',
            'N3IGV': 'IGV',
            'N3SCV': 'SCV',
            'N3HOT': 'HOT',
            'N3LOT': 'LOT'
        }, inplace=True)

        # 2. 기본 전처리 (날짜 변환 및 이름 통일)
        df_data['DATE'] = pd.to_datetime(df_data['DATE'])
        maint['NR_REQUEST_DATE'] = pd.to_datetime(maint['NR_REQUEST_DATE'], errors='coerce')
        
        # 3. 시간 관련 특성 생성
        df_data = df_data.sort_values(by=['AC_NO', 'DATE'])
        df_data['hour'] = df_data['DATE'].dt.hour
        df_data['month'] = df_data['DATE'].dt.month
        df_data['dayofweek'] = df_data['DATE'].dt.dayofweek
        df_data['month_sin'] = np.sin(2 * np.pi * df_data['month'] / 12)
        df_data['month_cos'] = np.cos(2 * np.pi * df_data['month'] / 12)

        # 4. MALFUNCTION 특성 생성
        maint_for_merge = maint[['AC_NO', 'NR_REQUEST_DATE', 'MALFUNCTION']].copy()
        maint_for_merge.rename(columns={'NR_REQUEST_DATE': 'DATE'}, inplace=True)
        maint_for_merge.dropna(subset=['MALFUNCTION'], inplace=True)
        maint_for_merge['MALFUNCTION'] = maint_for_merge['MALFUNCTION'].astype(str)

        maint_ata_pivot = maint_for_merge.pivot_table(
            index=['AC_NO', 'DATE'],
            columns='MALFUNCTION', 
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

        return df_processed, maint, ata_cols

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
df_processed, maint, ata_cols = preprocess_data()

# 데이터 로딩 성공 시에만 전체 앱 실행
if df_processed is not None:
    # --- 사이드바 UI 구성 ---
    st.sidebar.header('분석 옵션 선택')

    # 1. 분석할 항공기 선택
    available_tails = sorted(df_processed['AC_NO'].unique().astype(str))
    selected_tail = st.sidebar.selectbox('1. 분석할 항공기 선택:', available_tails, index=available_tails.index('HL8001'))
    
    # 2. 머신러닝 및 Raw 데이터 시각화 체크박스
    run_ml_prediction = st.sidebar.checkbox('머신러닝 예측 실행', value=False)
    visualize_raw_data = st.sidebar.checkbox('Raw 데이터 시각화', value=False)

    # 머신러닝 선택 시 관련 옵션 표시
    if run_ml_prediction:
        st.sidebar.markdown("---")
        st.sidebar.header('머신러닝 예측 설정')
        base_features = ['EGTA', 'GLA', 'WB', 'PT', 'P2A', 'LCOT', 'LCIT', 'IGV', 'SCV', 'HOT', 'LOT']
        selected_target = st.sidebar.selectbox('예측 타겟 변수 선택:', base_features, index=len(base_features)-1)
        available_features_for_ui = [f for f in base_features if f != selected_target]
        selected_base_features = st.sidebar.multiselect('학습 피처 선택:',
                                                        available_features_for_ui,
                                                        default=available_features_for_ui)

    # Raw 데이터 시각화 선택 시 관련 옵션 표시
    if visualize_raw_data:
        st.sidebar.markdown("---")
        st.sidebar.header('Raw 데이터 시각화 설정')
        base_features = ['EGTA', 'GLA', 'WB', 'PT', 'P2A', 'LCOT', 'LCIT', 'IGV', 'SCV', 'HOT', 'LOT']
        selected_raw_features = st.sidebar.multiselect(
            '시각화할 피처 선택:',
            base_features,
            default=['LOT','HOT']
        )

    # --- 분석 시작 버튼 ---
    if st.sidebar.button('분석 시작', type="primary") or 'analysis_done' in st.session_state:
        st.session_state['analysis_done'] = True
        
        if not run_ml_prediction and not visualize_raw_data:
            st.info("실행할 분석 옵션을 하나 이상 선택해주세요.")
            
        # --- 1. Raw 데이터 시각화 모드 ---
        if visualize_raw_data:
            st.subheader(f"📊 {selected_tail} - Raw 데이터 시각화")
            
            df_raw_plot = df_processed[df_processed['AC_NO'] == selected_tail].copy()
            
            if df_raw_plot.empty:
                st.warning(f"선택된 항공기({selected_tail})에 대한 데이터가 없습니다.")
            elif not selected_raw_features:
                st.warning("시각화할 피처를 하나 이상 선택해주세요.")
            else:
                fig_raw = go.Figure()
                for feature in selected_raw_features:
                    fig_raw.add_trace(go.Scatter(
                        x=df_raw_plot['DATE'],
                        y=df_raw_plot[feature],
                        mode='lines',
                        name=feature,
                        line={'width': 2},
                        hovertemplate='값: %{y:.2f}'
                    ))
                
                fig_raw.update_layout(
                    title=f'{selected_tail} - {", ".join(selected_raw_features)} 데이터 추이',
                    xaxis_title='날짜',
                    yaxis_title='값',
                    hovermode="x unified",
                    xaxis={'dtick': 'M3', 'tickformat': '%Y-%m-%d'}
                )
                st.plotly_chart(fig_raw, use_container_width=True)

                st.write(f"📈 {selected_tail}의 {', '.join(selected_raw_features)} 전체 기간 데이터")
                
                display_cols = ['DATE'] + selected_raw_features
                st.dataframe(df_raw_plot[display_cols], height=240, use_container_width=True)
        
            st.markdown("---")

        # --- 2. 머신러닝 모드 ---
        if run_ml_prediction:
            with st.spinner('통합 모델 학습 및 결과 분석 중...'):
                
                feature_names_dict = {
                    'EGTA': 'APU 배기 가스 온도', 'GLA': 'APU 발전기 부하',
                    'WB': '보정된 부하 압축기 공기 공급량', 'PT': '블리드 공기 압력',
                    'P2A': 'APU 흡입 압력', 'LCOT': '부하 압축기 출구 온도',
                    'LCIT': '부하 압축기 입구 온도', 'IGV': '흡입 공기 조절깃 위치',
                    'SCV': '압력 조절 밸브 위치', 'HOT': 'High Oil Temperature',
                    'LOT': 'Low Oil Temperature',
                }

                time_features = ['hour', 'month_sin', 'month_cos', 'dayofweek']
                selected_features = selected_base_features + time_features
                numerical_features = [f for f in selected_features if f in feature_names_dict or f in time_features]
                categorical_features = ata_cols
                all_model_features = numerical_features + categorical_features

                trained_pipeline, cv_scores = train_unified_model(df_processed, selected_target, numerical_features, categorical_features)
                
                df_processed['Predicted'] = trained_pipeline.predict(df_processed[all_model_features])
                df_processed['Residual'] = df_processed[selected_target] - df_processed['Predicted']
                df_tail_analysis = df_processed[df_processed['AC_NO'] == selected_tail].copy()
                
                df_plot = df_tail_analysis

                # --- 선택한 항공기 예측 및 분석 (전체 기간) ---
                st.subheader(f"1. {selected_tail} - {selected_target} 예측 분석")
                
                # === 수정된 부분: 토글 버튼 레이블 변경 ===
                show_details = st.toggle(
                    "자동 이상치 탐지 및 모델 세부 정보 표시", 
                    value=False, 
                    help="활성화하면 그래프에 예측 이상치를 표시하고, 하단에 이상치 목록과 모델 상세 정보를 함께 보여줍니다."
                )
                
                if df_plot.empty:
                    st.warning(f"{selected_tail}에 대한 데이터가 없습니다.")
                else:
                    model = IsolationForest(contamination=0.01, random_state=42)
                    model.fit(df_plot[[selected_target, 'Predicted']])
                    df_plot['outlier'] = model.fit_predict(df_plot[[selected_target, 'Predicted']])
                    outliers_df = df_plot[df_plot['outlier'] == -1].copy()
                    
                    unfiltered_outliers_exist = not outliers_df.empty 
                    plot_title_period = "전체 기간"
                    
                    # HL8001이 선택된 경우, 시각화할 기간을 24년 7월부터 25년 3월까지로 제한
                    if selected_tail == 'HL8001':
                        start_date = pd.to_datetime('2024-07-01')
                        end_date = pd.to_datetime('2025-03-31')
                        # 플롯 데이터 필터링
                        df_plot = df_plot[(df_plot['DATE'] >= start_date) & (df_plot['DATE'] <= end_date)].copy()
                        # 플롯과 목록에 표시될 이상치 데이터도 동일하게 필터링
                        outliers_df = outliers_df[(outliers_df['DATE'] >= start_date) & (outliers_df['DATE'] <= end_date)].copy()
                        plot_title_period = "2024년 7월 ~ 2025년 3월"

                    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
                    
                    maint_dates = maint[maint['AC_NO'] == selected_tail]['NR_REQUEST_DATE'].dropna()
                    
                    sns.lineplot(x='DATE', y=selected_target, data=df_plot, label=f'실제 {selected_target}', color='blue', ax=axes[0], marker='o', markersize=3, alpha=0.7, errorbar=None)
                    sns.lineplot(x='DATE', y='Predicted', data=df_plot, label=f'예측 {selected_target}', color='red', linestyle='--', ax=axes[0], errorbar=None)
                    
                    if show_details and not outliers_df.empty:
                        axes[0].scatter(x=outliers_df['DATE'], y=outliers_df[selected_target], color='red', s=100, marker='X', label='Isolation Forest 이상치', zorder=5)

                    for i, date in enumerate(maint_dates.unique()):
                        axes[0].axvline(x=date, color='gold', linestyle='-', linewidth=4, label='정비 기록' if i == 0 else "")
                    axes[0].set_title(f'{selected_tail} - {selected_target} 예측값 분석 ({plot_title_period})', fontsize=16)
                    axes[0].set_ylabel(f'{selected_target} 값')
                    axes[0].grid(True, linestyle='--', alpha=0.6)
                    axes[0].legend()

                    max_abs_residual = df_plot['Residual'].abs().max() if not df_plot.empty else 20
                    y_limit = max(20, max_abs_residual + 5)
                    axes[1].fill_between(df_plot['DATE'], df_plot['Residual'], 0, where=(df_plot['Residual'] > 0), color='red', alpha=0.3, label='과소 예측 (실제값 > 예측값)')
                    axes[1].fill_between(df_plot['DATE'], df_plot['Residual'], 0, where=(df_plot['Residual'] < 0), color='blue', alpha=0.3, label='과대 예측 (실제값 < 예측값)')
                    axes[1].axhline(y=0, color='gray', linestyle='--')
                    axes[1].set_ylim(-y_limit, y_limit)
                    
                    if show_details and not outliers_df.empty:
                        axes[1].scatter(x=outliers_df['DATE'], y=outliers_df['Residual'], color='red', s=100, marker='X', label='Isolation Forest 이상치', zorder=5)

                    for i, date in enumerate(maint_dates.unique()):
                        axes[1].axvline(x=date, color='gold', linestyle='-', linewidth=4, label='정비 기록' if i == 0 else "")

                    axes[1].set_title('예측 잔차 (Residuals)', fontsize=16)
                    axes[1].set_xlabel('날짜')
                    axes[1].set_ylabel('잔차 값')
                    axes[1].grid(True, linestyle='--', alpha=0.6)
                    axes[1].legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                # --- 정비 기록 (항상 표시) ---
                st.markdown("---")
                st.markdown(f"##### {selected_tail} 정비 기록 (전체 기간)")
                maint_records = maint[maint['AC_NO'] == selected_tail].copy()
                if not maint_records.empty:
                    maint_records['DATE_STR'] = maint_records['NR_REQUEST_DATE'].dt.strftime('%Y-%m-%d')
                    display_df = maint_records[['DATE_STR', 'MALFUNCTION', 'MALFUNCTION_ATA', 'CORRECTIVE_ACTION']].sort_values(by='DATE_STR', ascending=True)
                    display_df.index = np.arange(1, len(display_df) + 1)
                    st.dataframe(display_df)
                else:
                    st.info("해당 기간에 정비 기록이 없습니다.")
                
                if show_details:
                    # --- 이상치 목록 섹션 ---
                    st.markdown("---")
                    st.subheader(f"2. 자동으로 감지된 이상치 목록 ({selected_tail})")
                    if not outliers_df.empty:
                        display_outliers = outliers_df[['DATE', selected_target, 'Predicted', 'Residual']].copy()
                        display_outliers['DATE_STR'] = display_outliers['DATE'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        display_outliers.rename(columns={'DATE_STR': '날짜', selected_target: '실제값', 'Predicted': '예측값', 'Residual': '잔차'}, inplace=True)
                        display_outliers.index = np.arange(1, len(display_outliers) + 1)
                        st.dataframe(display_outliers[['날짜', '실제값', '예측값', '잔차']], height=240)
                    else:
                        if unfiltered_outliers_exist and selected_tail == 'HL8001':
                            st.info(f"표시된 기간({plot_title_period}) 내에서는 자동으로 감지된 이상치가 없습니다.")
                        else:
                            st.info("Isolation Forest 분석 결과, 이상치가 감지되지 않았습니다.")
                    
                    # --- 모델 관련 세부사항 섹션 ---
                    st.markdown("---")
                    st.subheader("3. 모델 관련 세부사항")
                    
                    model_col1, model_col2 = st.columns(2)
                    
                    with model_col1:
                        st.markdown("#### 3-1. 통합 모델 학습 및 검증")
                        st.write("시계열 교차검증(TimeSeriesSplit, n=5) 평균 성능:")
                        st.metric("R² Score", f"{np.mean(cv_scores['R2']):.3f}")
                        st.markdown("R² 점수는 **1에 가까울수록** 모델이 데이터를 잘 설명한다는 의미입니다.")
                    
                    with model_col2:
                        st.markdown("#### 3-2. 학습에 사용된 피처 중요도 분석")
                        
                        numerical_feature_names = numerical_features
                        onehot_encoder = trained_pipeline.named_steps['preprocessor'].named_transformers_['onehot_categorical']
                        categorical_feature_names = list(onehot_encoder.get_feature_names_out(categorical_features))
                        
                        def get_korean_name(feature):
                            if feature in feature_names_dict:
                                return f'{feature} ({feature_names_dict[feature]})'
                            elif feature.startswith('ATA_'):
                                return f'{feature} (정비 코드)'
                            else:
                                return feature
                        
                        all_feature_names = [get_korean_name(f) for f in numerical_feature_names] + [get_korean_name(f) for f in categorical_feature_names]
                        
                        coefficients = trained_pipeline.named_steps['regressor'].coef_
                        
                        feature_importance_df = pd.DataFrame({
                            'Feature': all_feature_names,
                            'Coefficient': coefficients
                        })
                        
                        feature_importance_df = feature_importance_df[feature_importance_df['Coefficient'] != 0].copy()
                        feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
                        feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)
                        
                        if not feature_importance_df.empty:
                            fig_importance, ax = plt.subplots(figsize=(5, len(feature_importance_df) * 0.5))
                            heatmap_data = feature_importance_df[['Coefficient']].set_index(feature_importance_df['Feature'])
                            
                            sns.heatmap(
                                heatmap_data, 
                                annot=True, 
                                fmt=".2f", 
                                cmap='coolwarm', 
                                center=0,
                                cbar_kws={'label': '회귀 계수'},
                                ax=ax
                            )
                            ax.set_ylabel('')
                            ax.set_xlabel('회귀 계수')
                            ax.set_title(f'통합 모델 피처 중요도', fontsize=16)
                            plt.yticks(rotation=0)
                            plt.tight_layout()
                            st.pyplot(fig_importance)
                            
                            st.markdown(f"""
                            **회귀 계수 해석:**
                            - **양수(+)** 계수: 피처 값이 증가할수록 타겟 변수($$ {selected_target} $$) 값도 증가하는 경향이 있습니다. (정비례)
                            - **음수(-)** 계수: 피처 값이 증가할수록 타겟 변수($$ {selected_target} $$) 값은 감소하는 경향이 있습니다. (반비례)
                            - **절댓값**이 클수록 모델의 예측에 미치는 영향이 크다는 것을 의미합니다.
                            """)
                        else:
                            st.info("Lasso 모델의 페널티 설정으로 인해 모든 피처의 계수가 0이 되었습니다.")
