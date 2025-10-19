import chroma_db
import ask_query
import tkinter as tk
from tkinter import filedialog
import os

print('Welcome to Camel\n')
input('Press ENTER to continue.')
while True:
    os.system('cls')
    print('1. Add Data')
    print('2. Ask Query')
    print('3. Exit')
    choice = input('Enter your choice: ')

    if choice == '1':
        os.system('cls')
        print('1. Add PDF')
        print('2. Exit')
        choice = input('Enter your choice: ')
        if choice == '1':
            # Open file explorer
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            pdf_path = filedialog.askopenfilename(
                title="Select a PDF file",
                filetypes=[("PDF Files", "*.pdf")]
            )

            if not pdf_path:  # User cancelled
                print("No file selected.")
            else:
                collection_name = input('Enter collection name: ')
                print('Processing PDF...')
                chroma_db.add_chunks_to_vc(pdf_path, collection_name)
                input('Press Enter to continue...')
        elif choice == '2':
            break
    elif choice == '2':
        os.system('cls')
        collection_name = input('Enter collection name: ')
        graph = ask_query.build_rag_graph(collection_name)
        while True:
            print('Type "EXIT" to exit the program')
            query = input('Enter query: ')
            if query.lower() == 'exit':
                break
            elif query.strip() == '':
                print('Enter a valid query.')
            else:
                # Prepare initial state for StateGraph
                initial_state = {
                    "query": query,
                    "collection_name": collection_name,
                    "docs": [],
                    "answer": ""
                }
                # Invoke the StateGraph
                result = graph.invoke(initial_state)
                print("💡 Answer:", result["answer"])
    elif choice == '3':
        os.system('cls')
        print('Thank you for using Camel')
        print('Exiting')
        break









