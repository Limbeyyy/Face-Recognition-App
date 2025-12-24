package com.example.facedetapplication

import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import com.example.facedetapplication.db.AppDatabase

@Suppress("DEPRECATION")
class FaceActivity : AppCompatActivity() {

    private lateinit var nameInput: EditText
    private lateinit var btnAddGallery: Button
    private lateinit var btnSubmit: Button
    private val selectedUris = mutableListOf<Uri>()

    private lateinit var pipeline: RecognitionPipeline

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_face)

        nameInput = findViewById(R.id.name_input)
        btnAddGallery = findViewById(R.id.btn_add_gallery)
        btnSubmit = findViewById(R.id.btn_submit)

        // Initialize pipeline
        pipeline = RecognitionPipeline(
            FaceDetector(this),
            LandmarkExtractor(this),
            FaceRecognition(this),
            AntiSpoof(this, "spoof_model_scale_2_7.tflite"),
            VectorStorage(AppDatabase.create(this)),
            TemporalAggregator()
        )

        btnAddGallery.setOnClickListener {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "image/*"
                putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            }
            startActivityForResult(intent, 101)
        }


        btnSubmit.setOnClickListener {
            val name = nameInput.text.toString()
            val enrollActivity = EnrollActivity() // create instance
            for (uri in selectedUris) {
                val inputStream = contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()
                enrollActivity.enroll(name, bitmap)
            }
            finish()
        }

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101 && resultCode == RESULT_OK) {
            selectedUris.clear()
            data?.let { intent ->
                if (intent.clipData != null) {
                    val count = intent.clipData!!.itemCount
                    for (i in 0 until count) {
                        selectedUris.add(intent.clipData!!.getItemAt(i).uri)
                    }
                } else if (intent.data != null) {
                    selectedUris.add(intent.data!!)
                }
            }
            if (selectedUris.isNotEmpty()) {
                btnAddGallery.visibility = Button.GONE
                btnSubmit.visibility = Button.VISIBLE
            }
        }
    }
}
