package handlers

import (
	"archive/zip"
	"encoding/json"
	"golang_files/database"
	"golang_files/models"
	"html/template"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

var Tmpl *template.Template

func NewFile(w http.ResponseWriter, r *http.Request) {
	Tmpl.ExecuteTemplate(w, "new.html", nil)
}

func CreateFile(w http.ResponseWriter, r *http.Request) {
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "File upload failed", http.StatusBadRequest)
		return
	}
	defer file.Close()

	filename := header.Filename
	dst := "uploads/" + filename
	out, err := os.Create(dst)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer out.Close()

	_, err = io.Copy(out, file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	name := r.FormValue("name")
	if name == "" {
		name = filename
	}

	recordedAtStr := r.FormValue("recorded_at")
	recordedAt := time.Now()
	if recordedAtStr != "" {
		if parsed, err := time.Parse("2006-01-02T15:04", recordedAtStr); err == nil {
			recordedAt = parsed
		}
	}

	fileModel := models.File{
		Name:       name,
		Path:       dst,
		Size:       header.Size,
		RecordedAt: recordedAt,
	}

	if err := database.DB.Create(&fileModel).Error; err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	http.Redirect(w, r, "/files", http.StatusFound)
}

func GetFiles(w http.ResponseWriter, r *http.Request) {
	var files []models.File
	if err := database.DB.Find(&files).Error; err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	Tmpl.ExecuteTemplate(w, "files.html", map[string]interface{}{"files": files})
}

func GetFile(w http.ResponseWriter, r *http.Request, id uint) {
	var file models.File
	if err := database.DB.First(&file, id).Error; err != nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(file)
}

func UpdateFile(w http.ResponseWriter, r *http.Request, id uint) {
	var file models.File
	if err := database.DB.First(&file, id).Error; err != nil {
		http.NotFound(w, r)
		return
	}
	if err := json.NewDecoder(r.Body).Decode(&file); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := database.DB.Save(&file).Error; err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(file)
}

func DeleteFile(w http.ResponseWriter, r *http.Request, id uint) {
	var file models.File
	if err := database.DB.First(&file, id).Error; err != nil {
		http.NotFound(w, r)
		return
	}
	if err := database.DB.Delete(&file).Error; err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "File deleted"})
}

func ImportFile(w http.ResponseWriter, r *http.Request) {
	Tmpl.ExecuteTemplate(w, "import.html", nil)
}

func CreateImport(w http.ResponseWriter, r *http.Request) {
	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "File upload failed", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Save uploaded file temporarily
	tempFile, err := os.CreateTemp("", "import_*.zip")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	tempFile.Close() // Close the file
	defer os.Remove(tempFile.Name())

	out, err := os.Create(tempFile.Name())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer out.Close()

	_, err = io.Copy(out, file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Open ZIP
	rZip, err := zip.OpenReader(tempFile.Name())
	if err != nil {
		http.Error(w, "Invalid ZIP file", http.StatusBadRequest)
		return
	}
	defer rZip.Close()

	// Read manifest.json
	// manifest := make(map[string]map[string]interface{})
	// for _, f := range rZip.File {
	// 	if f.Name == "manifest.json" {
	// 		rc, err := f.Open()
	// 		if err != nil {
	// 			continue
	// 		}
	// 		err = json.NewDecoder(rc).Decode(&manifest)
	// 		rc.Close()
	// 		if err != nil {
	// 			// Log error but continue
	// 		}
	// 		break
	// 	}
	// }

	// Extract files
	for _, f := range rZip.File {
		if f.FileInfo().IsDir() || f.Name == "manifest.json" {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			continue
		}

		dst := filepath.Join("uploads", f.Name)
		if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
			rc.Close()
			continue
		}

		out, err := os.Create(dst)
		if err != nil {
			rc.Close()
			continue
		}

		_, err = io.Copy(out, rc)
		out.Close()
		rc.Close()
		if err != nil {
			continue
		}

		// Extract duration for videos
		duration := int64(0)
		// if strings.HasSuffix(strings.ToLower(f.Name), ".mp4") {
		// 	cmd := exec.Command("ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", dst)
		// 	var out bytes.Buffer
		// 	cmd.Stdout = &out
		// 	if err := cmd.Run(); err == nil {
		// 		var result map[string]interface{}
		// 		if err := json.Unmarshal(out.Bytes(), &result); err == nil {
		// 			if format, ok := result["format"].(map[string]interface{}); ok {
		// 				if durStr, ok := format["duration"].(string); ok {
		// 					if dur, err := strconv.ParseFloat(durStr, 64); err == nil {
		// 						duration = int64(dur)
		// 					}
		// 				}
		// 			}
		// 		}
		// 	}
		// }

		// Create file record
		fileModel := models.File{
			Name:       f.Name,
			Path:       dst,
			Size:       f.FileInfo().Size(),
			RecordedAt: time.Now(),
			Duration:   duration,
		}
		// data, ok := manifest[f.Name]
		// if !ok {
		// 	http.Error(w, "File "+f.Name+" not found in manifest", http.StatusBadRequest)
		// 	return
		// }
		// if uuid, ok := data["uuid"].(string); ok {
		// 	fileModel.UUID = uuid
		// }
		// if recordedAtVal, ok := data["recorded_at"]; ok {
		// 	switch v := recordedAtVal.(type) {
		// 	case string:
		// 		parsed, err := time.Parse(time.RFC3339Nano, v)
		// 		if err != nil {
		// 			http.Error(w, "Invalid recorded_at string format in manifest for file "+f.Name+": "+v, http.StatusBadRequest)
		// 			return
		// 		}
		// 		fileModel.RecordedAt = parsed
		// 	case float64:
		// 		fileModel.RecordedAt = time.Unix(int64(v), 0)
		// 	}
		// }
		// if durVal, ok := data["duration"]; ok {
		// 	if d, ok := durVal.(float64); ok {
		// 		fileModel.Duration = int64(d)
		// 	}
		// }
		// if uuid, ok := data["uuid"].(string); ok {
		// 	fileModel.UUID = uuid
		// }
		// log.Print(data)
		// if recordedAtVal, ok := data["recorded_at"]; ok {
		// 	switch v := recordedAtVal.(type) {
		// 	case string:
		// 		parsed, err := time.Parse(time.RFC3339Nano, v)
		// 		if err != nil {
		// 			http.Error(w, "Invalid recorded_at string format in manifest for file "+f.Name+": "+v, http.StatusBadRequest)
		// 			return
		// 		}
		// 		fileModel.RecordedAt = parsed
		// 	case float64:
		// 		fileModel.RecordedAt = time.Unix(int64(v), 0)
		// 	}
		// }


		if err := database.DB.Create(&fileModel).Error; err != nil {
			// Log error but continue
		}
	}

	http.Redirect(w, r, "/files", http.StatusFound)
}

func Player(w http.ResponseWriter, r *http.Request) {
	Tmpl.ExecuteTemplate(w, "player.html", nil)
}