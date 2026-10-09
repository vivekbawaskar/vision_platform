-- Real-Time Object Detection & Logging Platform: MySQL setup script.
-- Run manually (mysql -u root -p < database_schema.sql) or let
-- database.py create everything automatically on first start.

CREATE DATABASE IF NOT EXISTS vision_platform
    CHARACTER SET utf8mb4;

USE vision_platform;

CREATE TABLE IF NOT EXISTS event_logs (
    log_id       INT AUTO_INCREMENT PRIMARY KEY,
    timestamp    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    object_class VARCHAR(50) NOT NULL,
    confidence   FLOAT NOT NULL,
    bbox_x       FLOAT NOT NULL,
    bbox_y       FLOAT NOT NULL,
    bbox_w       FLOAT NOT NULL,
    bbox_h       FLOAT NOT NULL,
    INDEX idx_timestamp (timestamp),
    INDEX idx_object_class (object_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
