"""
Restaurant Recommendation API & Frontend
Flask application serving recommendations and analytics.
"""

import os
import requests as http_requests
from datetime import datetime, timezone

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)
CORS(app)

# ─── Config ──────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "restaurant_reviews")

POSTGRES_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST",     "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname":   os.getenv("POSTGRES_DB",       "review_warehouse"),
    "user":     os.getenv("POSTGRES_USER",     "pipeline_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "secure_password"),
}

AIRFLOW_BASE = os.getenv("AIRFLOW_BASE_URL",  "http://localhost:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER",      "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASSWORD",  "admin")


# ─── DB helpers ──────────────────────────────────────────────
def get_mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client[MONGO_DB]


def get_postgres():
    return psycopg2.connect(**POSTGRES_CONFIG, cursor_factory=RealDictCursor,
                            connect_timeout=5)


# ─── Frontend ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─── Health ──────────────────────────────────────────────────
@app.route("/api/v1/health")
def health_check():
    try:
        db = get_mongo()
        stats = {
            col: db[col].count_documents({})
            for col in ["raw_reviews", "restaurants",
                        "processed_reviews", "recommendations"]
        }
        status = "healthy"
    except Exception as e:
        stats = {}
        status = f"degraded: {e}"

    return jsonify({
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collections": stats,
    })


# ─── Restaurants ─────────────────────────────────────────────
@app.route("/api/v1/restaurants")
def list_restaurants():
    db = get_mongo()
    restaurants = list(
        db["restaurants"].find({}, {"_id": 0}).sort("rating", -1).limit(100)
    )
    return jsonify({"count": len(restaurants), "restaurants": restaurants})


@app.route("/api/v1/restaurants/<place_id>")
def get_restaurant(place_id):
    db = get_mongo()
    restaurant = db["restaurants"].find_one({"place_id": place_id}, {"_id": 0})
    if not restaurant:
        return jsonify({"error": "Restaurant not found"}), 404
    return jsonify(restaurant)


@app.route("/api/v1/restaurants/<place_id>/reviews")
def get_restaurant_reviews(place_id):
    db = get_mongo()
    reviews = list(
        db["processed_reviews"]
        .find({"place_id": place_id}, {"_id": 0})
        .sort("predicted_score", -1)
        .limit(50)
    )
    # Fall back to raw reviews if no processed ones yet
    if not reviews:
        reviews = list(
            db["raw_reviews"]
            .find({"place_id": place_id}, {"_id": 0})
            .sort("rating", -1)
            .limit(50)
        )
    return jsonify({"count": len(reviews), "reviews": reviews})


# ─── Restaurant Search (the main search endpoint) ────────────
@app.route("/api/v1/restaurants/search")
def search_restaurants():
    """
    Search restaurants from MongoDB with optional filters.
    Returns results immediately from existing data — no pipeline run needed.

    Query params:
        q          – partial restaurant name (case-insensitive)
        min_rating – minimum avg Google rating (float)
        min_nlp    – minimum avg NLP predicted score (float)
        cuisine    – cuisine keyword matched against 'types'
        limit      – max results (default 20)
    """
    db = get_mongo()

    q          = request.args.get("q", "").strip()
    min_rating = request.args.get("min_rating", type=float)
    min_nlp    = request.args.get("min_nlp",    type=float)
    cuisine    = request.args.get("cuisine",    "").strip().lower()
    limit      = request.args.get("limit",  default=20, type=int)

    # Build MongoDB filter
    mongo_filter = {}
    if q:
        mongo_filter["name"] = {"$regex": q, "$options": "i"}
    if min_rating:
        mongo_filter["rating"] = {"$gte": min_rating}
    if cuisine:
        mongo_filter["types"] = {"$elemMatch": {"$regex": cuisine, "$options": "i"}}

    restaurants = list(
        db["restaurants"]
        .find(mongo_filter, {"_id": 0})
        .sort("rating", -1)
        .limit(limit)
    )

    if not restaurants:
        return jsonify({"count": 0, "restaurants": [],
                        "message": "No restaurants found. Try adjusting filters or trigger the pipeline."})

    # Enrich with latest NLP scores from processed_reviews
    place_ids = [r["place_id"] for r in restaurants]
    pipeline = [
        {"$match": {"place_id": {"$in": place_ids}}},
        {"$group": {
            "_id": "$place_id",
            "avg_nlp_score":    {"$avg": "$predicted_score"},
            "positive_count":   {"$sum": {"$cond": [{"$in": ["$sentiment_label",
                                    ["positive", "very_positive"]]}, 1, 0]}},
            "negative_count":   {"$sum": {"$cond": [{"$in": ["$sentiment_label",
                                    ["negative", "very_negative"]]}, 1, 0]}},
            "review_count":     {"$sum": 1},
        }},
    ]
    nlp_by_place = {
        doc["_id"]: doc
        for doc in db["processed_reviews"].aggregate(pipeline)
    }

    # Merge NLP stats into restaurant docs
    enriched = []
    for r in restaurants:
        nlp = nlp_by_place.get(r["place_id"], {})
        n   = nlp.get("review_count", 0)
        pos = nlp.get("positive_count", 0)
        r["avg_nlp_score"]  = round(nlp.get("avg_nlp_score", 0) or 0, 2)
        r["positive_ratio"] = round(pos / n, 3) if n else 0
        r["nlp_review_count"] = n

        # Apply optional NLP score filter
        if min_nlp and r["avg_nlp_score"] < min_nlp:
            continue
        enriched.append(r)

    return jsonify({"count": len(enriched), "restaurants": enriched})


# ─── Recommendations ─────────────────────────────────────────
@app.route("/api/v1/recommendations")
def get_recommendations():
    """
    Returns ranked recommendations, personalized when a user_id is supplied.

    Query params:
        user_id    – look up saved preferences for this user (optional)
        min_rating – override minimum avg_predicted_score (optional)
        cuisine    – override cuisine type filter (optional)
        limit      – max results (default 20)
    """
    db = get_mongo()

    user_id    = request.args.get("user_id", "").strip()
    min_rating = request.args.get("min_rating", type=float)
    limit      = request.args.get("limit", default=20, type=int)

    # cuisine param accepts comma-separated values: ?cuisine=italian,sushi
    raw_cuisine = request.args.get("cuisine", "").strip().lower()
    cuisines = [c.strip() for c in raw_cuisine.split(",") if c.strip()] if raw_cuisine else []

    # Load saved user preferences; query params take priority over stored prefs
    if user_id:
        pref_doc = db["user_preferences"].find_one({"user_id": user_id}, {"_id": 0})
        if pref_doc:
            prefs = pref_doc.get("preferences", {})
            if min_rating is None and prefs.get("min_rating"):
                min_rating = prefs["min_rating"]
            if not cuisines and prefs.get("cuisine_types"):
                cuisines = [c.lower() for c in prefs["cuisine_types"]]

    rec_doc = db["recommendations"].find_one({"user_id": "global"}, {"_id": 0})

    if rec_doc:
        recommendations = rec_doc.get("recommendations", [])

        if min_rating:
            recommendations = [
                r for r in recommendations
                if r.get("avg_predicted_score", 0) >= min_rating
            ]

        # Enrich with types + apply multi-cuisine OR-match filter in one pass
        place_ids = [r["place_id"] for r in recommendations]
        rest_meta = {
            doc["place_id"]: doc
            for doc in db["restaurants"].find(
                {"place_id": {"$in": place_ids}}, {"place_id": 1, "types": 1, "rating": 1}
            )
        }
        enriched = []
        for r in recommendations:
            meta  = rest_meta.get(r["place_id"], {})
            types = meta.get("types", [])
            if cuisines:
                types_lower = [t.lower() for t in types]
                if not any(
                    preferred in restaurant_type
                    for preferred in cuisines
                    for restaurant_type in types_lower
                ):
                    continue
            enriched.append({**r, "types": types, "google_rating": meta.get("rating")})

        return jsonify({
            "count": len(enriched[:limit]),
            "recommendations": enriched[:limit],
            "generated_at": rec_doc.get("generated_at"),
            "source": "precomputed",
            "personalized_for": user_id or None,
            "active_filters": {
                "cuisines": cuisines,
                "min_rating": min_rating,
            },
        })

    # ── Fallback: pipeline hasn't run yet — build live from MongoDB ──
    pipeline = [
        {"$group": {
            "_id": "$place_id",
            "avg_nlp_score":  {"$avg": "$predicted_score"},
            "review_count":   {"$sum": 1},
            "pos_count":      {"$sum": {"$cond": [{"$in": ["$sentiment_label",
                                ["positive", "very_positive"]]}, 1, 0]}},
        }},
        {"$match": {"review_count": {"$gte": 1}}},
        {"$sort": {"avg_nlp_score": -1}},
        {"$limit": limit},
    ]
    nlp_docs = list(db["processed_reviews"].aggregate(pipeline))

    if not nlp_docs:
        # Absolute fallback: just return restaurants sorted by Google rating
        restaurants = list(
            db["restaurants"].find({}, {"_id": 0}).sort("rating", -1).limit(limit)
        )
        return jsonify({
            "count": len(restaurants),
            "recommendations": [
                {
                    "rank": i + 1,
                    "place_id": r.get("place_id"),
                    "restaurant_name": r.get("name"),
                    "avg_predicted_score": r.get("rating", 0),
                    "recommendation_score": r.get("rating", 0),
                    "review_count": r.get("total_ratings", 0),
                    "positive_ratio": 0,
                    "sentiment_summary": {},
                }
                for i, r in enumerate(restaurants)
            ],
            "generated_at": None,
            "source": "google_rating_fallback",
            "message": "NLP pipeline has not run yet — showing Google ratings.",
        })

    # Enrich NLP results with restaurant metadata
    place_ids   = [d["_id"] for d in nlp_docs]
    restaurants = {
        r["place_id"]: r
        for r in db["restaurants"].find(
            {"place_id": {"$in": place_ids}}, {"_id": 0}
        )
    }

    recommendations = []
    for i, doc in enumerate(nlp_docs):
        pid  = doc["_id"]
        rest = restaurants.get(pid, {})
        n    = doc["review_count"]
        recommendations.append({
            "rank": i + 1,
            "place_id": pid,
            "restaurant_name": rest.get("name", "Unknown"),
            "avg_predicted_score": round(doc["avg_nlp_score"] or 0, 2),
            "recommendation_score": round(doc["avg_nlp_score"] or 0, 2),
            "review_count": n,
            "positive_ratio": round(doc["pos_count"] / n, 3) if n else 0,
            "sentiment_summary": {},
        })

    return jsonify({
        "count": len(recommendations),
        "recommendations": recommendations,
        "generated_at": None,
        "source": "live_nlp",
        "message": "Recommendation pipeline not yet run — showing live NLP averages.",
    })


# ─── Pipeline Trigger ────────────────────────────────────────
@app.route("/api/v1/pipeline/trigger", methods=["POST"])
def trigger_pipeline():
    """
    Trigger the Airflow review_ingestion_pipeline DAG.

    POST body (JSON):
        lat           – GPS latitude  (optional, defaults to Tempe)
        lng           – GPS longitude (optional, defaults to Tempe)
        location_name – human-readable label (optional)
    """
    body = request.get_json(silent=True) or {}
    lat  = body.get("lat")
    lng  = body.get("lng")
    name = body.get("location_name", "Tempe")

    dag_conf = {"location_name": name}
    if lat is not None and lng is not None:
        dag_conf["user_lat"]  = lat
        dag_conf["user_lng"]  = lng

    run_id = f"frontend_trigger_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    payload = {"dag_run_id": run_id, "conf": dag_conf}

    try:
        resp = http_requests.post(
            f"{AIRFLOW_BASE}/api/v1/dags/review_ingestion_pipeline/dagRuns",
            json=payload,
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=10,
        )
        if resp.status_code in (200, 409):      # 409 = already running, still OK
            return jsonify({
                "status": "triggered",
                "dag_run_id": run_id,
                "location": dag_conf,
            })
        return jsonify({
            "status": "error",
            "detail": resp.text,
        }), resp.status_code

    except http_requests.exceptions.ConnectionError:
        return jsonify({
            "status": "error",
            "detail": "Cannot reach Airflow. Is it running on port 8080?",
        }), 503


# ─── Analytics ───────────────────────────────────────────────
@app.route("/api/v1/analytics/sentiment")
def sentiment_analytics():
    """Sentiment distribution analytics from the PostgreSQL warehouse."""
    try:
        conn = get_postgres()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    r.name,
                    r.place_id,
                    COUNT(p.review_id)                                     AS prediction_count,
                    ROUND(AVG(p.predicted_score)::numeric, 2)              AS avg_nlp_score,
                    ROUND(AVG(f.rating)::numeric,          2)              AS avg_actual_rating,
                    SUM(CASE WHEN p.sentiment_label IN ('positive','very_positive')
                             THEN 1 ELSE 0 END)                            AS positive_count,
                    SUM(CASE WHEN p.sentiment_label IN ('negative','very_negative')
                             THEN 1 ELSE 0 END)                            AS negative_count
                FROM fact_predictions  p
                JOIN fact_reviews      f ON p.review_id = f.review_id
                JOIN dim_restaurants   r ON p.place_id  = r.place_id
                GROUP BY r.name, r.place_id
                HAVING COUNT(p.review_id) >= 3
                ORDER BY avg_nlp_score DESC
                LIMIT 50
            """)
            results = [dict(row) for row in cur.fetchall()]
        conn.close()
        return jsonify({"count": len(results), "analytics": results})
    except Exception as e:
        return jsonify({"error": str(e), "analytics": []}), 500


# ─── Preferences ─────────────────────────────────────────────
@app.route("/api/v1/preferences", methods=["POST"])
def submit_preferences():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No preferences provided"}), 400

    user_id     = data.get("user_id", "anonymous")
    preferences = {
        "cuisine_types": data.get("cuisine_types", []),
        "min_rating":    data.get("min_rating", 3.0),
        "price_range":   data.get("price_range"),
    }

    db = get_mongo()
    db["user_preferences"].update_one(
        {"user_id": user_id},
        {"$set": {"preferences": preferences,
                  "updated_at": datetime.now(timezone.utc).isoformat()}},
        upsert=True,
    )
    return jsonify({"status": "saved", "user_id": user_id, "preferences": preferences})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
