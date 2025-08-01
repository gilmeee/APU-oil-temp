import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Matplotlib 한글 폰트 설정 (필요 시, 맞는 폰트 이름으로 변경)
# from matplotlib import font_manager, rc
# font_path = "c:/Windows/Fonts/malgun.ttf"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide", page_title="APU 성능 예측 대시보드")
st.title('✈️ APU 성능 예측 인터랙티브 대시보드')

# --- 데이터 로딩 및 전처리 함수 (캐싱으로 속도 향상) ---
@st.cache_data
def load_data():
    """데이터를 로드하고 기본적인 전처리를 수행합니다."""
    try:
        df_data = pd.read_csv('APU_2023-06-29_2025-07-30_processedver2.csv')
        too_high = pd.read_csv('APU_high_temp_maint.csv')
        maint = pd.read_csv('APU_maint.csv')

        df_data.dropna(how='all', axis=1, inplace=True)
        df_data['CREATION_DATE'] = pd.to_datetime(df_data['CREATION_DATE'])
        # 원본 코드의 N3LOT 기반 dropna를 유지하되, 다른 타겟을 선택할 경우를 대비해 유연성 고려
        df_data.dropna(subset=['N3LOT'], inplace=True)

        too_high['NR_REQUEST_DATE'] = pd.to_datetime(too_high['NR_REQUEST_DATE'], errors='coerce')
        maint['NR_REQUEST_DATE'] = pd.to_datetime(maint['NR_REQUEST_DATE'], errors='coerce')
        
        return df_data, too_high, maint
    except FileNotFoundError as e:
        st.error(f"오류: '{e.filename}' 파일을 찾을 수 없습니다. CSV 파일들이 스크립트와 같은 폴더에 있는지 확인해주세요.")
        return None, None, None

# '정상' 데이터 마스크 함수
def get_normal_data_mask(df, fault_df, maint_df, tail_num, window_days=7):
    fault_dates = pd.to_datetime(fault_df[fault_df['AC_NO'] == tail_num]['NR_REQUEST_DATE'], errors='coerce').dropna()
    maint_dates = pd.to_datetime(maint_df[maint_df['AC_NO'] == tail_num]['NR_REQUEST_DATE'], errors='coerce').dropna()
    all_event_dates = pd.to_datetime(pd.concat([fault_dates, maint_dates]).unique(), errors='coerce').dropna()
    
    is_normal_data = pd.Series(True, index=df.index)
    for event_date in all_event_dates:
        start_exclusion = event_date - pd.Timedelta(days=window_days)
        end_exclusion = event_date
        is_normal_data &= ~((df['CREATION_DATE'] >= start_exclusion) & (df['CREATION_DATE'] <= end_exclusion))
    return is_normal_data

# --- 데이터 로드 ---
df_data, too_high, maint = load_data()

# 데이터 로딩 성공 시에만 전체 앱 실행
if df_data is not None:
    # --- 사이드바 UI 구성 ---
    st.sidebar.header('⚙️ 모델 파라미터 선택')

    all_features = ['N3EGTA', 'N3GLA', 'N3WB', 'N3PT', 'N3P2A', 'N3LCOT', 'N3LCIT', 'N3IGV', 'N3SCV', 'N3HOT']
    all_targets =  ['N3EGTA', 'N3GLA', 'N3WB', 'N3PT', 'N3P2A', 'N3LCOT', 'N3LCIT', 'N3IGV', 'N3SCV', 'N3HOT']
    available_tails = sorted(df_data['REGNO'].unique().astype(str))

    selected_tail = st.sidebar.selectbox('1. 항공기 기번 선택:', available_tails, index=available_tails.index('HL8001'))
    selected_target = st.sidebar.selectbox('2. 예측 타겟 변수 선택:', all_targets)
    
    # 타겟으로 선택된 변수는 피처 목록에서 자동 제외
    available_features = [f for f in all_features if f != selected_target]
    selected_features = st.sidebar.multiselect('3. 학습 피처 변수 선택:', available_features, default=available_features)

    # --- 분석 시작 버튼 ---
    if st.sidebar.button('📊 분석 시작', type="primary"):
        with st.spinner('모델 학습 및 시각화 진행 중...'):
            
            # --- 원본 코드의 핵심 로직 시작 ---
            
            # 1. 사용자가 선택한 값으로 변수 설정
            tail = selected_tail
            features = selected_features
            target = selected_target

            # 2. 데이터 분리
            train_end_date = pd.to_datetime('2024-12-31')
            df_train = df_data[df_data['CREATION_DATE'] <= train_end_date].copy()
            df_test = df_data[df_data['CREATION_DATE'] > train_end_date].copy()

            df_train_tail = df_train[df_train['REGNO'] == tail].copy().sort_values(by='CREATION_DATE')
            df_test_tail = df_test[df_test['REGNO'] == tail].copy().sort_values(by='CREATION_DATE')

            if len(df_train_tail) < 30:
                st.warning(f"항공기 {tail}의 학습 데이터가 부족하여 분석을 중단합니다.")
            else:
                st.success(f"항공기: {tail} 분석을 시작합니다.")
                
                # 3. 모델 학습
                normal_mask = get_normal_data_mask(df_train_tail, too_high, maint, tail, window_days=7)
                X_train_full = df_train_tail[features]
                y_train_full = df_train_tail[target]
                X_train_normal = X_train_full[normal_mask]
                y_train_normal = y_train_full[normal_mask]
                
                if len(X_train_normal) < 30:
                    st.warning("정비/고장 이력 제외 후 학습 데이터가 부족합니다.")
                else:
                    preprocessor = ColumnTransformer([('scaler', StandardScaler(), features)], remainder='drop')
                    pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', Lasso(max_iter=10000))])
                    tscv = TimeSeriesSplit(n_splits=5)
                    param_grid = {'regressor__alpha': [0.001, 0.01, 0.1, 1, 10]}
                    search = GridSearchCV(pipeline, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
                    search.fit(X_train_normal, y_train_normal)
                    best_model = search.best_estimator_

                    st.write(f"**모델 학습 완료!** (최적 Alpha: `{search.best_params_['regressor__alpha']}`)")

                    # 4. 예측 및 잔차 계산
                    df_train_tail['Predicted'] = best_model.predict(X_train_full)
                    df_train_tail['Residual'] = df_train_tail[target] - df_train_tail['Predicted']

                    # 5. 학습 기간 시각화 및 정비 기록 출력
                    st.markdown("---")
                    st.subheader('1. 학습 기간(Training Period) 분석')

                    # 학습 기간 정비 기록 처리
                    maint_records_train_period = maint[(maint['AC_NO'] == tail) & (maint['NR_REQUEST_DATE'] >= df_train_tail['CREATION_DATE'].min()) & (maint['NR_REQUEST_DATE'] <= df_train_tail['CREATION_DATE'].max())].copy()
                    fault_dates_train = too_high[(too_high['AC_NO'] == tail) & (too_high['NR_REQUEST_DATE'] >= df_train_tail['CREATION_DATE'].min()) & (too_high['NR_REQUEST_DATE'] <= df_train_tail['CREATION_DATE'].max())]['NR_REQUEST_DATE'].dropna()

                    # 시각화 1: 학습 기간
                    fig1, axes1 = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
                    sns.lineplot(x='CREATION_DATE', y=target, data=df_train_tail, label=f'Actual {target}', color='blue', ax=axes1[0], marker='o', markersize=3)
                    sns.lineplot(x='CREATION_DATE', y='Predicted', data=df_train_tail, label=f'Predicted {target}', color='red', linestyle='--', ax=axes1[0])
                    for i, date in enumerate(fault_dates_train):
                        axes1[0].axvline(x=date, color='black', linestyle='-', linewidth=1.5, label='High Temp Fault' if i == 0 else "")
                    
                    # (코드 중략) 원본과 동일한 정비 기록 시각화 로직
                    plotted_maint_labels_train = {'Maint. Record': False, 'Special Maint.': False}
                    for date in maint_records_train_period['NR_REQUEST_DATE'].dropna().unique():
                        daily_maint_data = maint_records_train_period[maint_records_train_period['NR_REQUEST_DATE'] == date]
                        malfunction_text = " ".join(daily_maint_data['MALFUNCTION'].dropna().astype(str).str.upper())
                        action_text = " ".join(daily_maint_data['CORRECTIVE_ACTION'].dropna().astype(str).str.upper())
                        line_color, line_label = ('orange', 'Maint. Record')
                        if 'ODOR' in malfunction_text or 'CLEAN' in action_text:
                            line_color, line_label = ('green', 'Special Maint.')
                        if not plotted_maint_labels_train[line_label]:
                            axes1[0].axvline(x=date, color=line_color, linestyle='-', linewidth=1.5, label=line_label)
                            plotted_maint_labels_train[line_label] = True
                        else:
                            axes1[0].axvline(x=date, color=line_color, linestyle='-', linewidth=1.5)
                    
                    axes1[0].set_title(f'{target}: Actual vs. Prediction (Aircraft: {tail}, Training Period)'); axes1[0].set_ylabel(f'{target} Value'); axes1[0].grid(True)
                    handles, labels = axes1[0].get_legend_handles_labels()
                    axes1[0].legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
                    axes1[1].fill_between(df_train_tail['CREATION_DATE'], df_train_tail['Residual'], 0, where=(df_train_tail['Residual'] > 0), color='red', alpha=0.3, label='Positive Residual (Model Under-predicts)')
                    axes1[1].fill_between(df_train_tail['CREATION_DATE'], df_train_tail['Residual'], 0, where=(df_train_tail['Residual'] < 0), color='blue', alpha=0.3, label='Negative Residual (Model Over-predicts)')
                    axes1[1].axhline(y=0, color='gray', linestyle='--'); axes1[1].set_title(f'Residuals of {target} Prediction'); axes1[1].set_xlabel('Date'); axes1[1].set_ylabel('Residual Value'); axes1[1].legend(); axes1[1].grid(True)
                    plt.tight_layout()
                    st.pyplot(fig1)

                    # 학습 기간 정비 기록 테이블 출력
                    st.write("#### 학습 기간 정비 기록")
                    if not maint_records_train_period.empty:
                        maint_records_train_period['DATE_STR'] = maint_records_train_period['NR_REQUEST_DATE'].dt.strftime('%Y-%m-%d')
                        grouped_maint = maint_records_train_period.groupby('DATE_STR').agg({'MALFUNCTION': lambda x: '; '.join(x.dropna().astype(str).unique()), 'CORRECTIVE_ACTION': lambda x: '; '.join(x.dropna().astype(str).unique())}).reset_index()
                        st.dataframe(grouped_maint)
                    else:
                        st.info("해당 기간에 정비 기록이 없습니다.")

                    # 6. 예측 기간 분석
                    if not df_test_tail.empty:
                        st.markdown("---")
                        st.subheader('2. 예측 기간(Test Period) 분석')
                        
                        df_test_tail['Predicted'] = best_model.predict(df_test_tail[features])
                        df_test_tail['Residual'] = df_test_tail[target] - df_test_tail['Predicted']

                        maint_records_test_period = maint[(maint['AC_NO'] == tail) & (maint['NR_REQUEST_DATE'] >= df_test_tail['CREATION_DATE'].min()) & (maint['NR_REQUEST_DATE'] <= df_test_tail['CREATION_DATE'].max())].copy()
                        fault_dates_test = too_high[(too_high['AC_NO'] == tail) & (too_high['NR_REQUEST_DATE'] >= df_test_tail['CREATION_DATE'].min()) & (too_high['NR_REQUEST_DATE'] <= df_test_tail['CREATION_DATE'].max())]['NR_REQUEST_DATE'].dropna()

                        # 시각화 2: 예측 기간
                        fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
                        sns.lineplot(x='CREATION_DATE', y=target, data=df_test_tail, label=f'Actual {target}', color='blue', ax=axes2[0], marker='o', markersize=3)
                        sns.lineplot(x='CREATION_DATE', y='Predicted', data=df_test_tail, label=f'Predicted {target}', color='red', linestyle='--', ax=axes2[0])
                        # (코드 중략) 원본과 동일한 정비 기록 시각화 로직
                        for i, date in enumerate(fault_dates_test):
                            axes2[0].axvline(x=date, color='black', linestyle='-', linewidth=1.5, label='High Temp Fault' if i == 0 else "")
                        
                        plotted_maint_labels_test = {'Maint. Record': False, 'Special Maint.': False}
                        for date in maint_records_test_period['NR_REQUEST_DATE'].dropna().unique():
                            daily_maint_data = maint_records_test_period[maint_records_test_period['NR_REQUEST_DATE'] == date]
                            malfunction_text = " ".join(daily_maint_data['MALFUNCTION'].dropna().astype(str).str.upper())
                            action_text = " ".join(daily_maint_data['CORRECTIVE_ACTION'].dropna().astype(str).str.upper())
                            line_color, line_label = ('orange', 'Maint. Record')
                            if 'ODOR' in malfunction_text or 'CLEAN' in action_text:
                                line_color, line_label = ('green', 'Special Maint.')
                            if not plotted_maint_labels_test[line_label]:
                                axes2[0].axvline(x=date, color=line_color, linestyle='-', linewidth=1.5, label=line_label)
                                plotted_maint_labels_test[line_label] = True
                            else:
                                axes2[0].axvline(x=date, color=line_color, linestyle='-', linewidth=1.5)

                        axes2[0].set_title(f'{target}: Actual vs. Prediction (Aircraft: {tail}, Prediction Period)'); axes2[0].set_ylabel(f'{target} Value'); axes2[0].grid(True)
                        handles, labels = axes2[0].get_legend_handles_labels()
                        axes2[0].legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
                        axes2[1].fill_between(df_test_tail['CREATION_DATE'], df_test_tail['Residual'], 0, where=(df_test_tail['Residual'] > 0), color='red', alpha=0.3, label='Positive Residual')
                        axes2[1].fill_between(df_test_tail['CREATION_DATE'], df_test_tail['Residual'], 0, where=(df_test_tail['Residual'] < 0), color='blue', alpha=0.3, label='Negative Residual')
                        axes2[1].axhline(y=0, color='gray', linestyle='--'); axes2[1].set_title(f'Residuals of {target} Prediction'); axes2[1].set_xlabel('Date'); axes2[1].set_ylabel('Residual Value'); axes2[1].legend(); axes2[1].grid(True)
                        plt.tight_layout()
                        st.pyplot(fig2)

                        # 예측 기간 정비 기록 테이블 출력
                        st.write("#### 예측 기간 정비 기록")
                        if not maint_records_test_period.empty:
                            maint_records_test_period['DATE_STR'] = maint_records_test_period['NR_REQUEST_DATE'].dt.strftime('%Y-%m-%d')
                            grouped_maint_test = maint_records_test_period.groupby('DATE_STR').agg({'MALFUNCTION': lambda x: '; '.join(x.dropna().astype(str).unique()), 'CORRECTIVE_ACTION': lambda x: '; '.join(x.dropna().astype(str).unique())}).reset_index()
                            st.dataframe(grouped_maint_test)
                        else:
                            st.info("해당 기간에 정비 기록이 없습니다.")
                    else:
                        st.info("해당 항공기는 예측 기간 데이터가 없습니다.")

