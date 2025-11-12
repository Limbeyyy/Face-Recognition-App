package com.example.attendancefacerecognition;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import com.example.attendancefacerecognition.UI.AttendanceActivity;
import com.example.attendancefacerecognition.UI.RegisterActivity;
import com.example.attendancefacerecognition.UI.DeviceLoginActivity;
import com.example.attendancefacerecognition.UI.AppPrefManager;

public class MainActivity extends AppCompatActivity {
    private Button registerBtn, attendanceBtn;
    private AppPrefManager appPref;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        appPref = new AppPrefManager(this);
        if (appPref.getDeviceId().isEmpty()) {
            startActivity(new Intent(this, DeviceLoginActivity.class));
            finish();
            return;
        }

        setContentView(R.layout.activity_main);
        registerBtn = findViewById(R.id.registerBtn);
        attendanceBtn = findViewById(R.id.attendanceBtn);

        registerBtn.setOnClickListener(v -> startActivity(new Intent(this, RegisterActivity.class)));
        attendanceBtn.setOnClickListener(v -> startActivity(new Intent(this, AttendanceActivity.class)));
    }
}
