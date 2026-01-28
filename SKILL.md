# Startup Knowledge RAG Skill

## Opis
Ovaj skill omogucava pristup bazi znanja o startupima, baziranoj na knjizi "The Startup Owner's Manual" i drugim resursima.

## Kada koristiti
Koristi ovaj MCP server kada korisnik pita o:
- Customer Discovery procesu
- Customer Validation strategijama
- Business Model Canvas-u
- MVP (Minimum Viable Product) razvoju
- Pivot odlukama
- Earlyvangelists identifikaciji
- Market Type analizi
- Startup metrikama
- Investicijama i fundraising-u
- Skaliranju startupa
- Bilo kojoj biznis strategiji za startupe

## MCP Endpoint
`POST http://localhost:8000/search`

## Request Format
```json
{
    "query": "Kako da validiram svoj startup?",
    "k": 5
}
```

## Response Format
```json
{
    "results": [
        {
            "text": "Relevantni tekst iz knowledge base...",
            "source": "startup_owners_manual.pdf",
            "relevance_score": 0.89
        }
    ],
    "total_results": 5
}
```

## Primjer koristenja

Korisnik: "Kako da nadjem prve kupce za moj B2B SaaS?"

1. Pozovi MCP sa query: "finding first customers B2B SaaS earlyvangelists"
2. Dobij relevantne chunk-ove iz knjige
3. Koristi te chunk-ove kao kontekst za odgovor
4. Citiraj izvor ako je relevantno
