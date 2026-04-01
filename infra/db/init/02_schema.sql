CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS movies (
    movie_id BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT NOT NULL,
    release_year INT
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id BIGINT NOT NULL,
    movie_id BIGINT NOT NULL,
    rating DOUBLE PRECISION NOT NULL,
    ts BIGINT NOT NULL,
    PRIMARY KEY (user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS tags (
    user_id BIGINT NOT NULL,
    movie_id BIGINT NOT NULL,
    tag TEXT NOT NULL,
    ts BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS movie_embeddings (
    movie_id BIGINT PRIMARY KEY REFERENCES movies(movie_id),
    embedding vector(256)
);

CREATE TABLE IF NOT EXISTS user_candidates (
    user_id BIGINT NOT NULL,
    movie_id BIGINT NOT NULL,
    retrieved_by_content BOOLEAN NOT NULL DEFAULT FALSE,
    retrieved_by_collaborative BOOLEAN NOT NULL DEFAULT FALSE,
    content_score DOUBLE PRECISION,
    content_rank INT,
    collaborative_score DOUBLE PRECISION,
    collaborative_rank INT,
    number_of_sources INT NOT NULL DEFAULT 1,
    PRIMARY KEY (user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS ranking_dataset (
    user_id BIGINT NOT NULL,
    movie_id BIGINT NOT NULL,
    label INT NOT NULL,
    split TEXT NOT NULL,
    features JSONB NOT NULL,
    PRIMARY KEY (user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    user_id BIGINT NOT NULL,
    movie_id BIGINT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    rank INT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_candidates_user ON user_candidates(user_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_user_rank ON recommendations(user_id, rank);
