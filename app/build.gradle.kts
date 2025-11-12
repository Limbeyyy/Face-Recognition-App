plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.attendancefacerecognition"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.attendancefacerecognition"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        vectorDrawables.useSupportLibrary = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false // Use '=' for boolean assignment
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions { // Add this block for Kotlin projects
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    // Use parentheses and double quotes for dependencies
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("com.google.android.material:material:1.11.0")

    // CameraX
    implementation("androidx.camera:camera-core:1.3.0")
    implementation("androidx.camera:camera-camera2:1.3.0")
    implementation("androidx.camera:camera-lifecycle:1.3.0")
    implementation("androidx.camera:camera-view:1.3.0")

    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.13.0")

    // JSON
    implementation("com.google.code.gson:gson:2.10.1")

    // MediaPipe (optional - add if you will implement MediaPipe detection)
    // implementation("com.google.mediapipe:mediapipe-face-detection:0.9.0")
}
