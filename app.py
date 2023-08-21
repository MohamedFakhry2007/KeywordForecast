from flask import Flask, render_template, request
import pandas as pd
from pytrends.request import TrendReq
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        term = request.form['search_term']
        country = request.form['country']  # Retrieve the country value here
        result = get_forecast(term, country)  # Pass both term and country to get_forecast
        return render_template('result.html', result=result, term=term, country=country)
    return render_template('index.html')


def get_forecast(term, country):
    country_codes = {
        'Afghanistan': 'AF', 'Albania': 'AL', 'Algeria': 'DZ', 'Andorra': 'AD', 'Angola': 'AO',
        'Antigua and Barbuda': 'AG',
        'Argentina': 'AR', 'Armenia': 'AM', 'Australia': 'AU', 'Austria': 'AT', 'Azerbaijan': 'AZ', 'Bahamas': 'BS',
        'Bahrain': 'BH', 'Bangladesh': 'BD', 'Barbados': 'BB', 'Belarus': 'BY', 'Belgium': 'BE', 'Belize': 'BZ',
        'Benin': 'BJ', 'Bhutan': 'BT', 'Bolivia': 'BO', 'Bosnia and Herzegovina': 'BA', 'Botswana': 'BW',
        'Brazil': 'BR',
        'Brunei': 'BN', 'Bulgaria': 'BG', 'Burkina Faso': 'BF', 'Burundi': 'BI', 'Cabo Verde': 'CV', 'Cambodia': 'KH',
        'Cameroon': 'CM', 'Canada': 'CA', 'Central African Republic': 'CF', 'Chad': 'TD', 'Chile': 'CL', 'China': 'CN',
        'Colombia': 'CO', 'Comoros': 'KM', 'Congo (Congo-Brazzaville)': 'CG', 'Costa Rica': 'CR', 'Croatia': 'HR',
        'Cuba': 'CU', 'Cyprus': 'CY', 'Czechia (Czech Republic)': 'CZ', 'Democratic Republic of the Congo': 'CD',
        'Egypt': 'EG',
        'Eritrea': 'ER', 'Estonia': 'EE', 'Eswatini (fmr. Swaziland)': 'SZ', 'Ethiopia': 'ET', 'Fiji': 'FJ',
        'Finland': 'FI',
        'France': 'FR', 'Gabon': 'GA', 'Gambia': 'GM', 'Georgia': 'GE', 'Germany': 'DE', 'Ghana': 'GH', 'Greece': 'GR',
        'Grenada': 'GD', 'Guatemala': 'GT', 'Guinea': 'GN', 'Guinea-Bissau': 'GW', 'Guyana': 'GY', 'Haiti': 'HT',
        'Holy See': 'VA', 'Honduras': 'HN', 'Hungary': 'HU', 'Iceland': 'IS', 'India': 'IN', 'Indonesia': 'ID',
        'Iran': 'IR',
        'Iraq': 'IQ', 'Ireland': 'IE', 'Israel': 'IL', 'Italy': 'IT', 'Jamaica': 'JM', 'Japan': 'JP', 'Jordan': 'JO',
        'Kazakhstan': 'KZ', 'Kenya': 'KE', 'Kiribati': 'KI', 'Kuwait': 'KW', 'Kyrgyzstan': 'KG', 'Laos': 'LA',
        'Latvia': 'LV',
        'Lebanon': 'LB', 'Lesotho': 'LS', 'Liberia': 'LR', 'Libya': 'LY', 'Liechtenstein': 'LI', 'Lithuania': 'LT',
        'Luxembourg': 'LU', 'Madagascar': 'MG', 'Malawi': 'MW', 'Malaysia': 'MY', 'Maldives': 'MV', 'Mali': 'ML',
        'Malta': 'MT', 'Marshall Islands': 'MH', 'Mauritania': 'MR', 'Mauritius': 'MU', 'Mexico': 'MX',
        'Micronesia': 'FM',
        'Moldova': 'MD', 'Monaco': 'MC', 'Mongolia': 'MN', 'Montenegro': 'ME', 'Morocco': 'MA', 'Mozambique': 'MZ',
        'Myanmar (formerly Burma)': 'MM', 'Namibia': 'NA', 'Nauru': 'NR', 'Nepal': 'NP', 'Netherlands': 'NL',
        'New Zealand': 'NZ',
        'Nicaragua': 'NI', 'Niger': 'NE', 'Nigeria': 'NG', 'North Korea': 'KP',
        'North Macedonia (formerly Macedonia)': 'MK',
        'Norway': 'NO', 'Oman': 'OM', 'Pakistan': 'PK', 'Palau': 'PW', 'Palestine State': 'PS', 'Panama': 'PA',
        'Papua New Guinea': 'PG', 'Paraguay': 'PY', 'Peru': 'PE', 'Philippines': 'PH', 'Poland': 'PL', 'Portugal': 'PT',
        'Qatar': 'QA', 'Romania': 'RO', 'Russia': 'RU', 'Rwanda': 'RW', 'Saint Kitts and Nevis': 'KN',
        'Saint Lucia': 'LC',
        'Saint Vincent and the Grenadines': 'VC', 'Samoa': 'WS', 'San Marino': 'SM', 'Sao Tome and Principe': 'ST',
        'Saudi Arabia': 'SA', 'Senegal': 'SN', 'Serbia': 'RS', 'Seychelles': 'SC', 'Sierra Leone': 'SL',
        'Singapore': 'SG',
        'Slovakia': 'SK', 'Slovenia': 'SI', 'Solomon Islands': 'SB', 'Somalia': 'SO', 'South Africa': 'ZA',
        'South Korea': 'KR',
        'South Sudan': 'SS', 'Spain': 'ES', 'Sri Lanka': 'LK', 'Sudan': 'SD', 'Suriname': 'SR', 'Sweden': 'SE',
        'Switzerland': 'CH', 'Syria': 'SY', 'Tajikistan': 'TJ', 'Tanzania': 'TZ', 'Thailand': 'TH', 'Timor-Leste': 'TL',
        'Togo': 'TG', 'Tonga': 'TO', 'Trinidad and Tobago': 'TT', 'Tunisia': 'TN', 'Turkey': 'TR', 'Turkmenistan': 'TM',
        'Tuvalu': 'TV', 'Uganda': 'UG', 'Ukraine': 'UA', 'United Arab Emirates': 'AE', 'United Kingdom': 'GB',
        'United States of America': 'US', 'Uruguay': 'UY', 'Uzbekistan': 'UZ', 'Vanuatu': 'VU', 'Venezuela': 'VE',
        'Vietnam': 'VN', 'Yemen': 'YE', 'Zambia': 'ZM', 'Zimbabwe': 'ZW'
    }

    geo = country_codes[country]
    pytrends = TrendReq(hl='en-US', tz=180)
    time_period = 'today 5-y'
    pytrends.build_payload([term], timeframe=time_period, geo=geo)
    iot_df = pytrends.interest_over_time()
    decomposition = seasonal_decompose(iot_df[term], model='additive', period=12)

    trend = decomposition.trend.dropna()
    model = auto_arima(trend, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                       trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)

    forecast = model.predict(n_periods=30)
    forecast_series = pd.Series(forecast,
                                index=pd.date_range(start=trend.index[-1] + pd.Timedelta(days=1), periods=30, freq='D'))

    seasonal = decomposition.seasonal
    seasonal_mean = seasonal.groupby(seasonal.index.month).mean()
    adjusted_forecast = forecast_series + seasonal_mean[forecast_series.index.month].values

    # Create a visual representation for the result
    plt.figure(figsize=(10, 5))
    plt.plot(iot_df[term], label='Historical interest over time')
    plt.plot(adjusted_forecast, label='Adjusted forecasted interest over time')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    if adjusted_forecast.mean() > iot_df[term].mean():
        forecast_result = f'The search for "{term}" is expected to increase next month.'
        forecast_explanation = 'This forecast is based on the predicted upward trend for the upcoming period.'
    elif adjusted_forecast.mean() < iot_df[term].mean():
        forecast_result = f'The search for "{term}" is expected to decrease next month.'
        forecast_explanation = 'This forecast is based on the predicted downward trend for the upcoming period.'
    else:
        forecast_result = f'No expected change in the search for "{term}" next month.'
        forecast_explanation = ''

    return {'forecast': forecast_result, 'explanation': forecast_explanation, 'plot_url': plot_url}


if __name__ == '__main__':
    import subprocess

    gunicorn_cmd = [
        'gunicorn',  # Path to gunicorn executable if necessary
        '-b', '0.0.0.0:8000',  # Bind address and port
        '--timeout', '160',  # Worker timeout in seconds
        'app:app'  # Your Flask app module
    ]

    subprocess.run(gunicorn_cmd)
