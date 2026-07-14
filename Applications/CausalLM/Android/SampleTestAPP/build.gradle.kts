plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.example.sampletestapp"
    compileSdk = 36

    packaging {
        jniLibs.useLegacyPackaging = true
    }
    defaultConfig {
        applicationId = "com.example.sampletestapp"
        minSdk = 33
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // SampleTestAPP hosts the QuickDotAI AAR directly (no remote
            // :remote process) so it packages the AAR's arm64-v8a
            // jniLibs. Restrict to the matching ABI to avoid empty
            // armv7/x86_64 slices.
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        // LiteRT-LM 0.10.0 (the version our mirror serves) was compiled
        // with Kotlin 2.3.0 but our compiler is 2.2.21; the flag tells
        // kotlinc to accept the newer metadata stamp on transitive
        // artifacts. See libs.versions.toml for the full story.
        freeCompilerArgs += "-Xskip-metadata-version-check"
    }
}

dependencies {
    // The whole point of SampleTestAPP: depend on the :QuickDotAI AAR
    // directly and drive the catalog-selected engine in-process, without
    // QuickAIService. This project dependency also supplies the library's
    // declared runtime dependencies. A standalone .aar file has no Gradle
    // dependency metadata; see QuickDotAI/README.md for that integration.
    implementation(project(":QuickDotAI"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.activity)
    implementation(libs.material)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
