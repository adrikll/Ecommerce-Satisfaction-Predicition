import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

def run_book_scraper_pipeline():
    print("Iniciando Módulo de Pipeline de Dados: Extração de books.toscrape.com...")
    
    base_url = "http://books.toscrape.com/catalogue/"
    current_url = "http://books.toscrape.com/catalogue/page-1.html"
    
    all_books = []
    page_count = 1

    while current_url:
        print(f"Extraindo dados da página {page_count}: {current_url}")
        
        try:
            response = requests.get(current_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao acessar a página: {e}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        books = soup.find_all('article', class_='product_pod')

        if not books:
            print("Nenhum livro encontrado. Fim da extração.")
            break

        for book in books:
            title = book.h3.a['title']
            price_str = book.find('p', class_='price_color').text.strip().replace('£', '')
            price = float(price_str)
            
            rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
            rating_class = book.find('p', class_='star-rating')['class'][1]
            rating = rating_map.get(rating_class, 0)
            
            category_url = book.h3.a['href']
            category_response = requests.get(base_url + category_url.replace('../', ''))
            category_soup = BeautifulSoup(category_response.content, 'html.parser')
            category = category_soup.find('ul', class_='breadcrumb').find_all('li')[2].a.text.strip()
            
            all_books.append({
                'titulo': title,
                'preco': price,
                'nota_avaliacao': rating,
                'categoria': category
            })

        next_page_tag = soup.find('li', class_='next')
        if next_page_tag and next_page_tag.a:
            current_url = base_url + next_page_tag.a['href']
            page_count += 1
            time.sleep(1)
        else:
            current_url = None

    if all_books:
        df = pd.DataFrame(all_books)
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", "livros_extraidos.csv")
        df.to_csv(output_path, index=False)
        print("-" * 50)
        print("Extração concluída com sucesso!")
        print(f"Total de {len(df)} livros extraídos.")
        print(f"Arquivo salvo em: {output_path}")
    else:
        print("Nenhum livro foi extraído.")

if __name__ == "__main__":
    run_book_scraper_pipeline()