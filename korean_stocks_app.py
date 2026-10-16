from datetime import date, timedelta
from email.message import EmailMessage
from html import escape
from io import BytesIO
from datetime import datetime
import smtplib
import zipfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from openai import OpenAI
from pypdf import PdfReader
import requests


st.set_page_config(page_title="국내 주식 대시보드", page_icon="📈", layout="wide")

STOCKS = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "삼성바이오로직스": "207940.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "셀트리온": "068270.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "LG에너지솔루션": "373220.KS",
    "POSCO홀딩스": "005490.KS",
}

NEWS_QUERIES = {
    "삼성전자": "Samsung Electronics",
    "SK하이닉스": "SK Hynix",
    "삼성바이오로직스": "Samsung Biologics",
    "현대차": "Hyundai Motor",
    "기아": "Kia Motors",
    "셀트리온": "Celltrion",
    "NAVER": "NAVER Korea",
    "카카오": "Kakao Korea",
    "LG에너지솔루션": "LG Energy Solution",
    "POSCO홀딩스": "POSCO Holdings",
}


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(period: str) -> pd.DataFrame:
    tickers = list(STOCKS.values())
    data = yf.download(tickers, period=period, auto_adjust=False, progress=False, threads=True)
    if data.empty:
        return pd.DataFrame()
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    close.columns = [next((name for name, code in STOCKS.items() if code == col), col) for col in close.columns]
    return close.dropna(how="all")


@st.cache_data(ttl=900, show_spinner=False)
def load_news() -> pd.DataFrame:
    rows = []
    for name in STOCKS:
        try:
            # 일부 국내 티커는 Ticker.news가 빈 목록을 반환하므로 Yahoo 검색 API를 사용합니다.
            search = yf.Search(NEWS_QUERIES[name], news_count=8)
            for item in search.news:
                content = item.get("content", item)
                title = content.get("title", item.get("title", ""))
                link_value = content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else ""
                link = link_value or item.get("link", "")
                provider_value = content.get("provider", {})
                provider = provider_value.get("displayName", "Yahoo Finance") if isinstance(provider_value, dict) else item.get("publisher", "Yahoo Finance")
                raw_date = content.get("pubDate", item.get("providerPublishTime", ""))
                if isinstance(raw_date, (int, float)):
                    raw_date = datetime.fromtimestamp(raw_date).strftime("%Y-%m-%d %H:%M")
                pub_date = raw_date
                summary = content.get("summary", item.get("description", ""))
                if title and link:
                    rows.append({"종목": name, "제목": title, "언론사": provider, "게시일": pub_date, "요약": summary, "링크": link})
        except Exception:
            continue
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.drop_duplicates(subset=["링크"]).sort_values("게시일", ascending=False)
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def load_corp_codes(dart_api_key: str) -> dict:
    response = requests.get(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": dart_api_key},
        timeout=30,
    )
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        xml_name = archive.namelist()[0]
        xml_data = archive.read(xml_name)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_data)
    return {
        item.findtext("stock_code", "").strip(): item.findtext("corp_code", "").strip()
        for item in root.findall("list")
        if item.findtext("stock_code", "").strip()
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_disclosures(dart_api_key: str, corp_codes: dict) -> pd.DataFrame:
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    rows = []
    for name, ticker in STOCKS.items():
        stock_code = ticker.split(".")[0]
        corp_code = corp_codes.get(stock_code)
        if not corp_code:
            continue
        try:
            response = requests.get(
                "https://opendart.fss.or.kr/api/list.json",
                params={
                    "crtfc_key": dart_api_key,
                    "corp_code": corp_code,
                    "bgn_de": start_date.strftime("%Y%m%d"),
                    "end_de": end_date.strftime("%Y%m%d"),
                    "page_count": 20,
                },
                timeout=30,
            )
            payload = response.json()
            if payload.get("status") not in ("000", "013"):
                continue
            for item in payload.get("list", []):
                rows.append({
                    "종목": name,
                    "접수일": item.get("rcept_dt", ""),
                    "보고서명": item.get("report_nm", ""),
                    "제출인": item.get("flr_nm", ""),
                    "공시구분": item.get("corp_cls", ""),
                    "원문": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item.get('rcept_no', '')}",
                })
        except (requests.RequestException, ValueError):
            continue
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("접수일", ascending=False).drop_duplicates(subset=["종목", "보고서명", "접수일"])
    return result


def format_krw(value: float) -> str:
    return f"{value:,.0f}원" if pd.notna(value) else "-"


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(f"[페이지 {i + 1}]\n{text}" for i, text in enumerate(pages)).strip()


def refresh_prices() -> None:
    load_prices.clear()


def refresh_news() -> None:
    load_news.clear()


def refresh_disclosures() -> None:
    load_corp_codes.clear()
    load_disclosures.clear()


def build_stock_report(
    prices: pd.DataFrame,
    table: pd.DataFrame,
    returns: pd.Series,
    news: pd.DataFrame,
    period_label: str,
) -> tuple[str, str]:
    """Create a plain-text and HTML report from the data visible in this app."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_date = prices.index[-1].strftime("%Y-%m-%d")
    ordered_returns = returns.sort_values(ascending=False)
    ranking_lines = [
        f"{rank}. {name}: {value:+.2f}%"
        for rank, (name, value) in enumerate(ordered_returns.items(), start=1)
    ]
    stock_lines = [
        f"- {row['종목']}: 현재가 {row['현재가']}, 전일 대비 {row['전일 대비']}, "
        f"{period_label} 수익률 {row[f'{period_label} 수익률']}"
        for _, row in table.iterrows()
    ]

    news_rows = news.head(20) if not news.empty else pd.DataFrame()
    news_lines = [
        f"- [{row['종목']}] {row['제목']} ({row['언론사']}, {row['게시일']})\n  {row['링크']}"
        for _, row in news_rows.iterrows()
    ]
    news_text = "\n".join(news_lines) if news_lines else "수집된 뉴스가 없습니다."
    text_report = f"""국내 주식 인사이트 보고서
생성 시각: {generated_at}
주가 기준일: {data_date}
조회 기간: {period_label}

[기간 수익률 순위]
{chr(10).join(ranking_lines)}

[종목별 현황]
{chr(10).join(stock_lines)}

[최근 기업 뉴스]
{news_text}

본 보고서는 Yahoo Finance와 Yahoo Finance 뉴스 검색으로 수집된 데이터를 요약한 정보 제공용 자료이며, 투자 권유가 아닙니다.
"""

    table_html = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in table.itertuples(index=False, name=None)
    )
    headers_html = "".join(f"<th>{escape(str(column))}</th>" for column in table.columns)
    ranking_html = "".join(
        f"<li><strong>{escape(str(name))}</strong>: {value:+.2f}%</li>"
        for name, value in ordered_returns.items()
    )
    news_html = "".join(
        f"<li><strong>[{escape(str(row['종목']))}]</strong> "
        f"<a href=\"{escape(str(row['링크']), quote=True)}\">{escape(str(row['제목']))}</a>"
        f"<br><small>{escape(str(row['언론사']))} · {escape(str(row['게시일']))}</small>"
        f"<br>{escape(str(row['요약']))[:500]}</li>"
        for _, row in news_rows.iterrows()
    ) or "<li>수집된 뉴스가 없습니다.</li>"
    html_report = f"""<!doctype html>
<html><body style="font-family:Arial, sans-serif; color:#172b4d; line-height:1.55;">
<h1>국내 주식 인사이트 보고서</h1>
<p>생성 시각: {generated_at}<br>주가 기준일: {data_date}<br>조회 기간: {escape(period_label)}</p>
<h2>기간 수익률 순위</h2><ol>{ranking_html}</ol>
<h2>종목별 현황</h2>
<table style="border-collapse:collapse"><thead><tr>{headers_html}</tr></thead><tbody>{table_html}</tbody></table>
<h2>최근 기업 뉴스</h2><ol>{news_html}</ol>
<p style="color:#5e6c84;font-size:12px">본 보고서는 Yahoo Finance와 Yahoo Finance 뉴스 검색으로 수집된 데이터를 요약한 정보 제공용 자료이며, 투자 권유가 아닙니다.</p>
</body></html>"""
    return text_report, html_report


def send_smtp_report(
    host: str,
    port: int,
    sender: str,
    app_password: str,
    recipient: str,
    subject: str,
    text_report: str,
    html_report: str,
    use_ssl: bool,
) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(text_report)
    message.add_alternative(html_report, subtype="html")

    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
            smtp.login(sender, app_password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(sender, app_password)
            smtp.send_message(message)


st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg, #102a43, #1f5f8b); color: white; margin-bottom: 1.2rem;}
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: .45rem 0 0; color: #d9efff;}
</style>
<div class="hero"><h1>국내 주식 인사이트</h1><p>yfinance로 수집한 KOSPI 대표 종목 10개의 가격 흐름과 수익률을 한눈에 확인하세요.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("조회 설정")
    period_label = st.selectbox("기간", ["1개월", "3개월", "6개월", "1년", "2년"], index=2)
    period_map = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y"}
    selected = st.multiselect("차트 종목", list(STOCKS), default=["삼성전자", "SK하이닉스", "NAVER"])
    st.divider()
    openai_api_key = st.text_input("OpenAI API 키", type="password", key="openai_api_key", help="키는 이 페이지의 세션에서만 사용되며 파일에 저장하지 않습니다.")
    pdf_file = st.file_uploader("분석할 PDF 업로드", type=["pdf"], help="텍스트가 포함된 PDF를 분석합니다.")
    dart_api_key = st.text_input("OpenDART API 키", type="password", key="dart_api_key", help="OpenDART 인증키는 이 페이지의 세션에서만 사용됩니다.")
    st.divider()
    st.caption("데이터 새로고침")
    st.button("가격 데이터 새로고침", on_click=refresh_prices, use_container_width=True)
    st.button("뉴스 새로고침", on_click=refresh_news, use_container_width=True)
    st.button("공시 새로고침", on_click=refresh_disclosures, use_container_width=True)
    st.caption("데이터 출처: Yahoo Finance (yfinance) · 15분 단위 캐시")

prices = load_prices(period_map[period_label])
if prices.empty:
    st.error("주가 데이터를 불러오지 못했습니다. 인터넷 연결과 yfinance 설치 상태를 확인하세요.")
    st.stop()

latest = prices.ffill().iloc[-1]
previous = prices.ffill().iloc[-2] if len(prices) > 1 else latest
returns = (latest / prices.ffill().iloc[0] - 1) * 100

best_name = returns.idxmax()
worst_name = returns.idxmin()
cols = st.columns(4)
cols[0].metric("분석 종목", f"{len(STOCKS)}개")
cols[1].metric("최고 수익률", best_name, f"{returns[best_name]:+.2f}%")
cols[2].metric("최저 수익률", worst_name, f"{returns[worst_name]:+.2f}%")
cols[3].metric("기준일", prices.index[-1].strftime("%Y-%m-%d"))

price_header_col, price_button_col = st.columns([5, 1])
with price_header_col:
    st.subheader("종목별 현황")
with price_button_col:
    st.button("새로고침", on_click=refresh_prices, use_container_width=True, key="price_inline_refresh")
table = pd.DataFrame({
    "종목": list(STOCKS),
    "현재가": [format_krw(latest[name]) for name in STOCKS],
    "전일 대비": [f"{(latest[name] / previous[name] - 1) * 100:+.2f}%" for name in STOCKS],
    f"{period_label} 수익률": [f"{returns[name]:+.2f}%" for name in STOCKS],
    "티커": list(STOCKS.values()),
})
st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("주가 추이")
chart_names = selected or list(STOCKS)[:3]
normalized = prices[chart_names].ffill().apply(lambda s: s / s.iloc[0] * 100)
fig = go.Figure()
for name in chart_names:
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[name], mode="lines", name=name))
fig.update_layout(height=430, template="plotly_white", hovermode="x unified", yaxis_title="기준일 = 100", xaxis_title=None, legend_title=None)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📰 기업 관련 뉴스")
news_header_col, news_button_col = st.columns([5, 1])
with news_button_col:
    st.button("새로고침", on_click=refresh_news, use_container_width=True, key="news_inline_refresh")
news = load_news()
if news.empty:
    st.info("현재 수집된 뉴스가 없습니다. 잠시 후 다시 시도해 주세요.")
else:
    news_filter = st.multiselect("뉴스 종목 필터", list(STOCKS), default=list(STOCKS))
    filtered_news = news[news["종목"].isin(news_filter)] if news_filter else news.iloc[0:0]
    for _, item in filtered_news.head(20).iterrows():
        title = str(item["제목"])
        meta = f"{item['종목']} · {item['언론사']} · {item['게시일']}"
        st.markdown(f"**[{title}]({item['링크']})**  \n<small>{meta}</small>", unsafe_allow_html=True)
        if item["요약"]:
            st.caption(str(item["요약"])[:300])

st.divider()
st.subheader("✉️ 주식·뉴스 보고서 이메일 발송")
st.caption("현재 화면에서 수집한 주가와 최근 뉴스로 보고서를 생성합니다. SMTP 정보와 앱 비밀번호는 이 세션에서만 사용되며 파일에 저장하지 않습니다.")

report_text, report_html = build_stock_report(prices, table, returns, news, period_label)
report_filename = f"korean_stock_report_{prices.index[-1].strftime('%Y%m%d')}.txt"
preview_col, download_col = st.columns([3, 1])
with preview_col:
    with st.expander("보고서 미리보기"):
        st.text(report_text)
with download_col:
    st.download_button(
        "텍스트 보고서 다운로드",
        report_text.encode("utf-8-sig"),
        report_filename,
        "text/plain",
        use_container_width=True,
    )

with st.form("smtp_report_form", clear_on_submit=False):
    smtp_col, account_col = st.columns(2)
    with smtp_col:
        smtp_host = st.text_input("SMTP 서버", value="smtp.gmail.com", help="예: Gmail은 smtp.gmail.com")
        smtp_port = st.number_input("SMTP 포트", min_value=1, max_value=65535, value=465, step=1)
        smtp_ssl = st.checkbox("SSL 사용 (Gmail 권장)", value=True, help="해제하면 STARTTLS로 연결합니다.")
    with account_col:
        smtp_sender = st.text_input("발신 계정 이메일", placeholder="name@example.com")
        smtp_password = st.text_input("앱 비밀번호", type="password", help="일반 계정 비밀번호가 아닌 메일 서비스에서 발급한 앱 비밀번호를 입력하세요.")
        smtp_recipient = st.text_input("수신 이메일", placeholder="recipient@example.com")
    smtp_subject = st.text_input(
        "메일 제목",
        value=f"국내 주식·뉴스 보고서 ({prices.index[-1].strftime('%Y-%m-%d')})",
    )
    send_report = st.form_submit_button("보고서 이메일 발송", type="primary", use_container_width=True)

if send_report:
    missing_fields = [
        label for label, value in {
            "SMTP 서버": smtp_host,
            "발신 계정 이메일": smtp_sender,
            "앱 비밀번호": smtp_password,
            "수신 이메일": smtp_recipient,
        }.items() if not value.strip()
    ]
    if missing_fields:
        st.error(f"다음 항목을 입력해 주세요: {', '.join(missing_fields)}")
    elif "@" not in smtp_sender or "@" not in smtp_recipient:
        st.error("발신 계정과 수신 이메일 주소 형식을 확인해 주세요.")
    else:
        try:
            with st.spinner("SMTP 서버로 보고서를 전송하는 중입니다..."):
                send_smtp_report(
                    smtp_host.strip(),
                    int(smtp_port),
                    smtp_sender.strip(),
                    smtp_password,
                    smtp_recipient.strip(),
                    smtp_subject.strip(),
                    report_text,
                    report_html,
                    smtp_ssl,
                )
            st.success(f"보고서를 {smtp_recipient.strip()}로 발송했습니다.")
        except (OSError, smtplib.SMTPException) as exc:
            st.error(f"메일 발송에 실패했습니다: {exc}")

st.subheader("📑 OpenDART 기업 공시")
disclosure_header_col, disclosure_button_col = st.columns([5, 1])
with disclosure_button_col:
    st.button("새로고침", on_click=refresh_disclosures, use_container_width=True, key="disclosure_inline_refresh")
disclosures = pd.DataFrame()
if not dart_api_key:
    st.info("사이드바에 OpenDART API 키를 입력하면 최근 180일 공시를 조회할 수 있습니다.")
else:
    try:
        corp_codes = load_corp_codes(dart_api_key)
        disclosures = load_disclosures(dart_api_key, corp_codes)
        if disclosures.empty:
            st.warning("조회된 공시가 없습니다. API 키 또는 조회 기간을 확인해 주세요.")
        else:
            disclosure_filter = st.multiselect("공시 종목 필터", list(STOCKS), default=list(STOCKS))
            visible_disclosures = disclosures[disclosures["종목"].isin(disclosure_filter)]
            st.dataframe(
                visible_disclosures.head(50),
                column_config={"원문": st.column_config.LinkColumn("원문", display_text="공시 원문")},
                hide_index=True,
                use_container_width=True,
            )
    except Exception as exc:
        st.error(f"OpenDART 공시를 불러오지 못했습니다: {exc}")

left, right = st.columns(2)
with left:
    st.subheader("기간 수익률 순위")
    ranking = returns.sort_values(ascending=True).to_frame("수익률")
    ranking["수익률"] = ranking["수익률"].round(2)
    st.bar_chart(ranking, horizontal=True, color="#2f80ed")
with right:
    st.subheader("데이터 내려받기")
    csv = prices.to_csv(index=True).encode("utf-8-sig")
    st.download_button("CSV 다운로드", csv, "korean_stock_prices.csv", "text/csv", use_container_width=True)
    st.info(f"최근 데이터: {prices.index[-1].strftime('%Y-%m-%d')} · Yahoo Finance 제공 데이터는 지연될 수 있습니다.")

st.divider()
st.subheader("💬 주식 데이터 챗봇")
st.caption("현재 화면에 조회된 주가 데이터만 바탕으로 답변합니다. 투자 권유가 아닌 정보 제공용입니다.")

pdf_text = ""
if pdf_file is not None:
    try:
        pdf_text = extract_pdf_text(pdf_file)
        if pdf_text:
            st.success(f"{pdf_file.name} 업로드 완료 · {len(pdf_text):,}자 추출")
            with st.expander("PDF 텍스트 미리보기"):
                st.text(pdf_text[:3000] + ("\n..." if len(pdf_text) > 3000 else ""))
        else:
            st.warning("PDF에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF라면 OCR 처리가 필요합니다.")
    except Exception as exc:
        st.error(f"PDF를 읽는 중 오류가 발생했습니다: {exc}")

if "stock_chat_messages" not in st.session_state:
    st.session_state.stock_chat_messages = []

for message in st.session_state.stock_chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("예: 최근 수익률이 가장 높은 종목은?"):
    st.session_state.stock_chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not openai_api_key:
            answer = "사이드바에 OpenAI API 키를 입력하면 주식 데이터에 대해 답변할 수 있습니다."
            st.warning(answer)
        else:
            context = table.to_string(index=False)
            pdf_context = pdf_text[:30000] if pdf_text else "업로드된 PDF가 없습니다."
            news_context = news.head(30)[["종목", "제목", "언론사", "게시일", "요약", "링크"]].to_string(index=False) if not news.empty else "수집된 뉴스가 없습니다."
            disclosure_context = disclosures.head(50).to_string(index=False) if not disclosures.empty else "수집된 공시가 없습니다."
            system_prompt = f"""당신은 국내 주식 데이터 분석 도우미입니다.
아래 표, 뉴스, OpenDART 공시, 업로드된 PDF 내용만 사용해 한국어로 간결하고 정확하게 답변하세요.
뉴스와 공시는 제공된 제목·요약·보고서명을 근거로 답하고, 자료에 없는 사실을 추측하지 마세요.
표·뉴스·공시·PDF에 없는 정보(미래 주가 등)는 알 수 없다고 말하세요.
수익률은 조회 기간 기준이며, 투자 조언이 아니라 데이터 요약임을 필요할 때 밝혀주세요.

현재 조회 기간: {period_label}
데이터 기준일: {prices.index[-1].strftime('%Y-%m-%d')}
주가 데이터:
{context}

업로드된 PDF 내용:
{pdf_context}

최근 기업 뉴스:
{news_context}

최근 OpenDART 공시:
{disclosure_context}
"""
            try:
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.stock_chat_messages,
                    ],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
            except Exception as exc:
                answer = f"OpenAI API 호출 중 오류가 발생했습니다: {exc}"
                st.error(answer)
    st.session_state.stock_chat_messages.append({"role": "assistant", "content": answer})
