-- MySQL script to create the datasets table and stored procedures
CREATE TABLE IF NOT EXISTS datasets (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  s3_key VARCHAR(512) NOT NULL,
  name VARCHAR(255),
  mime VARCHAR(128),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DELIMITER $$
CREATE PROCEDURE sp_dataset_create(IN p_key VARCHAR(512), IN p_name VARCHAR(255), IN p_mime VARCHAR(128))
BEGIN
  INSERT INTO datasets(s3_key,name,mime) VALUES (p_key,p_name,p_mime);
  SELECT LAST_INSERT_ID() AS id;
END$$

CREATE PROCEDURE sp_dataset_get(IN p_id BIGINT)
BEGIN
  SELECT * FROM datasets WHERE id = p_id;
END$$
DELIMITER ;