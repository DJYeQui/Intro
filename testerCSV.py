import PyPDF2

def read_pdf(file_path):
    try:
        # PDF dosyasını aç
        with open(file_path, 'rb') as pdf_file:
            # PDF Reader oluştur
            reader = PyPDF2.PdfReader(pdf_file)

            # Toplam sayfa sayısını yazdır
            print(f"PDF toplam sayfa sayısı: {len(reader.pages)}\n")

            # Her sayfanın içeriğini yazdır
            for page_number, page in enumerate(reader.pages):
                print(f"--- Sayfa {page_number + 1} ---\n")
                print(page.extract_text())
                print("\n" + "-" * 30)
    except FileNotFoundError:
        print("Belirtilen dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")


# Kullanım
pdf_path = "_UK_AI_Opportunities_Action_Plan—_1736861556.pdf"  # Buraya PDF dosyanızın yolunu yazın
read_pdf(pdf_path)