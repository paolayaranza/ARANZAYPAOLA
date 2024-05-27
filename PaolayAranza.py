import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


imagen = open("regresion-lineal-multiple.png", "rb").read()

st.image(imagen, use_column_width=True)

st.title("Regression Analysis :chart_with_upwards_trend:")
st.write("Paola Larios - 0233696")
st.write("Aranza Arellano - 0239887")

st.write("---")

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["Introducción", "Simple linear regression", "VIX vs Implied Volatility"])

with tab1:
    st.subheader("Introducción")
    st.write("Simple linear regression is a statistical technique that seeks to model the relationship between a variable called the dependent variable and one or more explanatory variables, also called independent variables. In its simplest form, simple linear regression involves only one independent variable to predict the dependent variable.")
    st.write("This technique is especially useful when we want to understand how a dependent variable changes in relation to an independent variable.")
    st.write("Simple linear regression seeks to find the straight line that best fits the available data, minimizing the vertical distance between the data points and the regression line. Once this relationship is established, we can use the model to predict future values of the dependent variable based on the values of the independent variable.")

with tab2:
    st.subheader("Univariate Regression")
    st.write("There are many other factors that determine the size of house prices, but it is not the only one.")
    st.write("If we use only one variable in a regression, it is called simple regression, while the")
    st.write("Regressions with more than one variable are called multivariate regressions, simple regressions are easier to understand")
    st.subheader("How to do a regression in Python?")
    
    # Lista de símbolos de acciones
    tickers = ['INTC', 'NVDA', 'AMD', 'TXN', 'MU', 'SOXX']

    # Obtener fechas de inicio y fin desde el usuario
    start_date = st.date_input("Fecha de inicio:", value=pd.to_datetime('2023-01-01'))
    end_date = st.date_input("Fecha de fin:", value=pd.to_datetime('2023-12-31'))

    # Descargar datos históricos para cada símbolo
    historical_data = {}
    for ticker in tickers:
        historical_data[ticker] = yf.download(ticker, start=start_date, end=end_date)

    # Crear DataFrame combinado
    df_combined = pd.DataFrame()
    for ticker in tickers:
        if not historical_data[ticker].empty:
            df_combined[ticker] = historical_data[ticker]['Adj Close']

    # Normalizar los datos utilizando la estandarización (z-score)
    scaler = StandardScaler()
    df_combined_standardized = pd.DataFrame(scaler.fit_transform(df_combined), columns=df_combined.columns, index=df_combined.index)

    # Mostrar los datos normalizados en una tabla
    st.write("Datos históricos normalizados:")
    st.write(df_combined_standardized)

    # Selección de la variable independiente
    columnas = [col for col in df_combined_standardized.columns if col != 'SOXX']

    # Permitir al usuario seleccionar la variable independiente
    variable_independiente = st.selectbox("Seleccione la variable independiente:", columnas)

    st.subheader("Ejecución de la Regresión")
    
    X = df_combined_standardized[variable_independiente]  # Variable independiente
    Y = df_combined_standardized['SOXX']  # Variable dependiente

    # Calcular los coeficientes de la regresión lineal
    coefficients = np.polyfit(X, Y, 1)
    m, b = coefficients

    # Crear la gráfica de dispersión con la línea de regresión lineal
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.plot(X, m * X + b, color='red')  # Agregar la línea de regresión lineal

    # Configurar los ejes y etiquetas
    ax.set_ylabel('SOXX')
    ax.set_xlabel(variable_independiente)
    ax.set_title(f'SOXX vs {variable_independiente}')

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)

    st.write("----")

    st.subheader("Estadísticas")
    
    st.subheader("How to run the regression?")
    st.subheader("Statistics")

    # Agregar una columna de unos a X para el término constante
    X1 = sm.add_constant(X)

    # Ajustar el modelo de regresión lineal
    reg = sm.OLS(Y, X1).fit()

    # Mostrar el resumen en Streamlit
    st.write(reg.summary())

    st.write("---")

    st.subheader("Alpha, Beta, R-square")

    slope, intercept, r_value, p_value, std_err = linregress(X, Y)

    st.subheader("Conclusions")

    st.write(f"The slope is: ${slope:,.2f}")

    st.write(
        f"<span style='color: blue;'>It means that for every increment of the {variable_independiente}, its price is expected to rise by {slope:,.2f}.</span>",
        unsafe_allow_html=True)

    st.write(f"The intercept is: ${intercept:,.2f}")

    st.write("<span style='color: blue;'>It is the point that intercepts the y axis, also known as alpha.</span>",
             unsafe_allow_html=True)

    st.write(f"The r_value is: {(r_value ** 2) * 100:,.2f}%")

    st.write(
        f"<span style='color: blue;'> This means that the independent variable in our regression can explain {(r_value ** 2) * 100:,.2f}% of the dependent variable Accion price.</span>",
        unsafe_allow_html=True)

    st.write("<span style='color: blue;'>If the value is greater than 30%, then the model has good explanatory power.</span>",
             unsafe_allow_html=True)

    st.write(f"The p_value is: ${p_value:,.4e}")

    st.write("<span style='color: blue;'>The p value explains the relationship between the variables, if they are statistically significant or not.</span>",
             unsafe_allow_html=True)

    st.write(f"The error is: ${std_err:,.6f}")

    st.write(
        f"<span style='color: blue;'>On average, we can expect the actual responses to be about {std_err:,.6f} away from the fitted regression line.</span>",
        unsafe_allow_html=True)

with tab3:

    # Encabezado principal
    st.header('Implied Volatility Analysis')

    # Cargar los datos desde la URL
    url = "https://research-watchlists.s3.amazonaws.com/df_UniversidadPanamericana_ohlc.csv"
    df = pd.read_csv(url)
    df['time'] = pd.to_datetime(df['time'])

    # Eliminar la columna "Unnamed" si existe
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Filtrar los datos del último año
    one_year_ago = datetime.datetime.now() - pd.DateOffset(years=1)
    df_last_year = df[df['time'] >= one_year_ago]

    # Seleccionar símbolos
    symbols = df_last_year['Symbol'].unique()
    selected_symbols = st.multiselect("Select Symbols:", symbols)

    # Obtener datos históricos del VIX desde Yahoo Finance
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="1y")
    vix_data.reset_index(inplace=True)
    vix_data['Date'] = vix_data['Date'].dt.strftime('%Y-%m-%d')

    # Inicializar la figura de Plotly
    fig = go.Figure()

    # Graficar la línea de tendencia para el VIX
    fig.add_trace(go.Scatter(
        x=vix_data['Date'],
        y=vix_data['Close'],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='VIX'
    ))

    # Variables para almacenar datos de regresión
    all_volatilities = []
    all_vix_values = []

    # Graficar la volatilidad anualizada de cada símbolo seleccionado
    for symbol in selected_symbols:
        symbol_data = df_last_year[df_last_year['Symbol'] == symbol]
        symbol_data.sort_values(by='time', ascending=True, inplace=True)
        symbol_data['time'] = symbol_data['time'].dt.strftime('%Y-%m-%d')
        symbol_data['annualized_volatility'] = symbol_data['impVolatility'] * np.sqrt(252)
        
        # Añadir traza a la figura de Plotly
        fig.add_trace(go.Scatter(
            x=symbol_data['time'],
            y=symbol_data['annualized_volatility'],
            mode='lines+markers',
            name=f"{symbol} Annualized Volatility"
        ))
        
        # Añadir datos para la regresión
        all_volatilities.extend(symbol_data['annualized_volatility'].values)
        all_vix_values.extend(vix_data['Close'].values[:len(symbol_data)])  # Asegurarse de que las longitudes coincidan

    # Mostrar la gráfica interactiva en Streamlit
    st.plotly_chart(fig)

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st
    from sklearn.metrics import r2_score

    # Ejecución de la regresión
    st.subheader("Ejecución de la Regresión")

    # Convertir listas a arrays de numpy y eliminar NaNs/infs
    X = np.array(all_volatilities)
    Y = np.array(all_vix_values)

    # Filtrar NaNs e infinitos
    valid_idx = np.isfinite(X) & np.isfinite(Y)
    X = X[valid_idx]
    Y = Y[valid_idx]

    # Aplicar la transformación logarítmica
    X_log = np.log(X)
    Y_log = np.log(Y)

    try:
        # Calcular los coeficientes de la regresión lineal en los datos logarítmicos
        coefficients = np.polyfit(X_log, Y_log, 1)
        m_log, b_log = coefficients

        # Predecir los valores de Y utilizando el modelo
        Y_log_pred = m_log * X_log + b_log

        # Calcular el R²
        r2_log = r2_score(Y_log, Y_log_pred)

        # Crear la gráfica de dispersión con la línea de regresión lineal
        fig, ax = plt.subplots()
        ax.scatter(X_log, Y_log, label='Datos')
        ax.plot(X_log, Y_log_pred, color='red', label=f'Regresión log-log: y = {m_log:.2f}x + {b_log:.2f}')
        
        # Configurar los ejes y etiquetas
        ax.set_ylabel('Log(VIX)')
        ax.set_xlabel('Log(Annualized Volatility)')
        ax.set_title('VIX vs Annualized Volatility (Log-Log)')
        ax.legend()
        
        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)

        # Transformar los coeficientes a la escala original
        alpha_original = np.exp(b_log)
        beta_original = m_log

        # Mostrar los coeficientes en Streamlit
        st.write(f"Coeficiente de la pendiente (beta): {beta_original:.2f}")
        st.write(f"Intersección con el eje Y (alpha, en la escala original): {alpha_original:.2f}")
        st.write(f"Coeficiente de determinación (R²): {r2_log:.2f}")
        
        # Calcular los residuos
        residuals_log = Y_log - Y_log_pred

        # Crear una figura para la distribución de los residuos
        fig, ax = plt.subplots()
        sns.histplot(residuals_log, kde=True, ax=ax)
        ax.set_title('Distribución de Probabilidad de los Residuos (Log-Log)')
        ax.set_xlabel('Residuos')
        ax.set_ylabel('Densidad')

        # Mostrar la gráfica de distribución de residuos en Streamlit
        st.pyplot(fig)
        
        # Mostrar estadísticas de los residuos

        st.write(f"Desviación estándar de los residuos: {np.std(residuals_log):.2f}")

        # Conclusiones dinámicas
        st.subheader("Conclusiones")
        st.write(f"""
        ### 

        1. **Transformación Logarítmica**:
           - Para realizar un análisis más robusto y adecuado de la relación entre la volatilidad anualizada y el VIX, aplicamos una transformación logarítmica a ambos conjuntos de datos. Esto nos permite manejar mejor las posibles diferencias de escala y la naturaleza exponencial de los datos financieros.

        2. **Resultados de la Regresión Log-Log**:
           - La pendiente de la regresión lineal (beta) en la escala logarítmica es aproximadamente {beta_original:.2f}. Esto sugiere que hay una relación positiva entre la volatilidad anualizada y el VIX. En otras palabras, a medida que la volatilidad anualizada aumenta, el VIX también tiende a aumentar, y lo hace en una proporción sublineal.
           - La intersección (alpha) en la escala logarítmica, una vez convertida de nuevo a la escala original, es aproximadamente {alpha_original:.2f}. Esto representa el valor base del VIX cuando la volatilidad anualizada es 1 en la escala original.

        3. **Coeficiente de Determinación (R²)**:
           - El coeficiente de determinación (R²) es aproximadamente {r2_log:.2f}. Esto indica que el {r2_log*100:.2f}% de la variabilidad en los valores del VIX puede explicarse por la variabilidad en la volatilidad anualizada. Aunque no es un ajuste perfecto, muestra una relación considerablemente fuerte entre estas dos variables.

        4. **Distribución de los Residuos**:
           - El análisis de la distribución de los residuos muestra que los residuos (las diferencias entre los valores observados y los valores predichos) siguen una distribución aproximadamente normal. Esto sugiere que el modelo de regresión es adecuado y que no hay patrones evidentes en los residuos que indicarían un mal ajuste del modelo.

        ### Implicaciones

        La relación positiva entre la volatilidad anualizada y el VIX, capturada en la regresión log-log, es consistente con la teoría financiera que sugiere que el VIX (índice de volatilidad) es un indicador de la expectativa de volatilidad futura del mercado. 

        - **Inversores y Analistas**: Pueden utilizar esta relación para anticipar cambios en el VIX basándose en las variaciones de la volatilidad anualizada, ajustando sus estrategias de inversión y cobertura en consecuencia.
        - **Toma de Decisiones**: Este análisis puede ser útil para la gestión de riesgos y la formulación de políticas, especialmente en entornos de alta volatilidad.

        ### Limitaciones

        - **Datos Logarítmicos**: La interpretación directa de los resultados debe tener en cuenta la naturaleza logarítmica de los datos.
        - **Modelo Lineal**: Aunque la relación capturada es fuerte, el modelo es lineal en el espacio log-log. Relaciones más complejas pueden no ser capturadas por este modelo.

        En resumen, la regresión log-log realizada muestra una relación significativa y positiva entre la volatilidad anualizada y el VIX, con una explicación razonable de la variabilidad observada en los datos. Este análisis proporciona una herramienta valiosa para entender y prever las dinámicas del mercado basadas en la volatilidad.
        """)

    except Exception as e:
        st.error(f"Error en la ejecución de la regresión: {e}")

    st.write("----")

