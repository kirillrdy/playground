package main

import (
	"golang_files/database"
	"golang_files/handlers"
	"html/template"
	"log"
	"net/http"
	"strconv"
	"strings"
)

func main() {
	database.InitDB()

	tmpl := template.Must(template.ParseGlob("templates/*"))
	handlers.Tmpl = tmpl

	http.Handle("/uploads/", http.StripPrefix("/uploads/", http.FileServer(http.Dir("./uploads"))))

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(`<html><body><h1>Welcome to File Manager</h1><a href="/files">View Files</a></body></html>`))
	})

	http.HandleFunc("/files", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" {
			handlers.CreateFile(w, r)
		} else if r.Method == "GET" {
			handlers.GetFiles(w, r)
		} else {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	http.HandleFunc("/files/", func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if strings.HasSuffix(path, "/new") {
			handlers.NewFile(w, r)
			return
		}
		if strings.HasSuffix(path, "/import") {
			if r.Method == "GET" {
				handlers.ImportFile(w, r)
			} else if r.Method == "POST" {
				handlers.CreateImport(w, r)
			} else {
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			}
			return
		}
		idStr := strings.TrimPrefix(path, "/files/")
		if idStr == "" {
			handlers.GetFiles(w, r)
			return
		}
		id, err := strconv.Atoi(idStr)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		switch r.Method {
		case "GET":
			handlers.GetFile(w, r, uint(id))
		case "PUT":
			handlers.UpdateFile(w, r, uint(id))
		case "DELETE":
			handlers.DeleteFile(w, r, uint(id))
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	http.HandleFunc("/player", handlers.Player)

	log.Fatal(http.ListenAndServe(":3000", nil))
}
