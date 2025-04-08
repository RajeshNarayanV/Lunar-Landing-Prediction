CREATE TABLE mission_data (
    id SERIAL PRIMARY KEY,
    Nation TEXT,
    Type TEXT,
    Mission_Duration FLOAT,
    Launch_Year INT,
    Arrival_Year INT,
    Outcome TEXT
);

SELECT * FROM mission_data;



