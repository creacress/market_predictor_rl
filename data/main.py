import datetime
import pandas as pd
import yfinance as yf

def download_data(ticker="SAF.PA", start="2010-01-01"):
    end = datetime.datetime.today().strftime('%Y-%m-%d')
    print(f"Téléchargement des données de {ticker} jusqu'à {end}...")
    data = yf.download(ticker, start=start, end=end)

    # Nettoyage
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
        data.columns = [str(col).strip().lower().replace(" ", "_") for col in data.columns]
    else:
        raise ValueError("Les données téléchargées ne sont pas valides. Vérifie le ticker ou la connexion.")
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.sort_values('date').dropna(subset=['date'])

    if 'close' in data.columns:
        data['close'] = pd.to_numeric(data['close'], errors='coerce')
        data['ma_5'] = data['close'].rolling(window=5).mean()
        data['ma_20'] = data['close'].rolling(window=20).mean()
        data['return_1d'] = data['close'].pct_change()

    data = data.dropna()

    csv_path = f"data/{ticker.replace('.', '_')}_historical.csv"
    parquet_path = f"data/{ticker.replace('.', '_')}_clean.parquet"
    data.to_csv(csv_path, index=False)
    data.to_parquet(parquet_path, index=False)

    print(f"Données sauvegardées dans {csv_path} et {parquet_path}")
    return data

if __name__ == "__main__":
    df = download_data()
    print(df.tail())