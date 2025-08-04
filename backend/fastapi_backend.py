#!/usr/bin/env python3

"""
FastAPI Backend for Web Browser Query Agent
Provides REST API endpoints with optimized Google scraping
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the optimized query agent
from query_agent import WebBrowserQueryAgent, QueryValidator, SimilarityEngine, StorageManager


# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True


class QueryResponse(BaseModel):
    query: str
    cached: bool
    similar_query: Optional[str] = None
    summary: str
    sources: Optional[List[Dict]] = None
    response_time: float
    timestamp: str
    selector_used: Optional[str] = None  # Track which selector worked


class StatsResponse(BaseModel):
    total_queries: int
    cache_hits: int
    unique_queries: int
    avg_response_time: float
    recent_queries: List[Dict]
    selector_performance: Optional[Dict] = None  # Include selector stats


class ValidationResponse(BaseModel):
    is_valid: bool
    reason: str


# FastAPI app initialization
app = FastAPI(
    title="Optimized Web Browser Query Agent API",
    description="Intelligent web search with optimized selector performance and caching",
    version="2.1.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global query agent instance
query_agent = None
storage_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize the query agent on startup"""
    global query_agent, storage_manager
    api_key = os.getenv("GOOGLE_API_KEY")
    query_agent = WebBrowserQueryAgent(api_key)
    storage_manager = StorageManager()
    print("🚀 Optimized Query Agent API started successfully!")
    if api_key:
        print("✓ Gemini API key configured for enhanced summarization")
    else:
        print("⚠️ No Gemini API key found - using fallback summarization")
    
    # Log initial selector performance
    selector_perf = query_agent.scraper.selector_performance
    print("📊 Initial selector performance tracking enabled")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Optimized Web Browser Query Agent API",
        "version": "2.1.0",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Smart selector optimization",
            "Multi-engine search (Google primary, Bing fallback)",
            "Intelligent query caching",
            "Similarity matching",
            "Gemini AI summarization",
            "Performance tracking"
        ]
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check with selector performance"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    try:
        if query_agent:
            # Test validator
            is_valid, _ = query_agent.validator.is_valid("test query")
            health_status["components"]["validator"] = "healthy"
            
            # Test storage
            stored_queries = storage_manager.get_stored_queries()
            health_status["components"]["similarity_engine"] = "healthy"
            health_status["components"]["storage"] = f"healthy ({len(stored_queries)} queries)"
            
            # Test scraper components and show selector performance
            health_status["components"]["web_scraper"] = "healthy (optimized selectors)"
            
            # Show top performing selectors
            best_selectors = query_agent.scraper.get_optimized_selectors()[:3]
            health_status["components"]["top_selectors"] = best_selectors
            
        else:
            health_status["components"]["query_agent"] = "not_initialized"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["query_agent"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    health_status["components"]["gemini_api"] = "configured" if api_key else "not_configured"
    
    return health_status


@app.post("/api/search", response_model=QueryResponse)
async def search_query(request: QueryRequest):
    """Main search endpoint with selector performance tracking"""
    if not query_agent:
        print(f"[{datetime.now().isoformat()}] ERROR: Query agent not initialized")
        raise HTTPException(status_code=503, detail="Query agent not initialized")
    
    start_time = datetime.now()
    print(f"[{start_time.isoformat()}] INFO: Received search request for query: '{request.query}' (use_cache={request.use_cache})")
    
    try:
        # Validate query
        is_valid, reason = query_agent.validator.is_valid(request.query)
        print(f"[{datetime.now().isoformat()}] INFO: Query validation result: is_valid={is_valid}, reason='{reason}'")
        if not is_valid:
            print(f"[{datetime.now().isoformat()}] WARNING: Invalid query rejected")
            raise HTTPException(status_code=400, detail=f"Invalid query: {reason}")

        # Check cache
        stored_queries = storage_manager.get_stored_queries()
        similar_query = query_agent.similarity_engine.find_similar_query(request.query, stored_queries)
        selector_used = None
        
        if similar_query and request.use_cache:
            print(f"[{datetime.now().isoformat()}] INFO: Cache hit - Similar query found: '{similar_query['query']}' (similarity: {similar_query['similarity_score']:.2f})")
            storage_manager.update_access_count(similar_query['id'])
            summary = similar_query['summary']
            is_cached = True
            similar_query_text = similar_query['query']
            
            # Try to get selector info from cached results
            if 'results' in similar_query and similar_query['results']:
                for result in similar_query['results']:
                    if 'selector_used' in result:
                        selector_used = result['selector_used']
                        break
        else:
            print(f"[{datetime.now().isoformat()}] INFO: No similar query found, proceeding with web search")
            
            # Show current selector performance before search
            optimized_selectors = query_agent.scraper.get_optimized_selectors()
            print(f"[{datetime.now().isoformat()}] INFO: Using optimized selector order: {optimized_selectors[:3]}...")
            
            # Perform web search
            results = await query_agent.scraper.search_and_scrape(request.query)
            print(f"[{datetime.now().isoformat()}] INFO: Scraped {len(results)} results")
            
            # Extract selector used from results
            if results and 'selector_used' in results[0]:
                selector_used = results[0]['selector_used']
                print(f"[{datetime.now().isoformat()}] INFO: Successful selector: {selector_used}")
            
            if not results:
                print(f"[{datetime.now().isoformat()}] WARNING: No search results found for query")
                return QueryResponse(
                    query=request.query,
                    cached=False,
                    summary="No search results found for your query. This might be due to search engine blocking or connectivity issues.",
                    response_time=(datetime.now() - start_time).total_seconds(),
                    timestamp=datetime.now().isoformat(),
                    selector_used=selector_used
                )

            # Summarize results
            summary = query_agent.processor.summarize_results(request.query, results)
            print(f"[{datetime.now().isoformat()}] INFO: Summarization completed")
            
            # Store results
            storage_manager.store_query_result(request.query, results, summary)
            print(f"[{datetime.now().isoformat()}] INFO: Results cached for query")
            is_cached = False
            similar_query_text = None

        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Prepare sources
        sources = []
        if not is_cached:
            try:
                recent_queries = storage_manager.get_stored_queries()
                if recent_queries and recent_queries[0]['query'].lower() == request.query.lower():
                    sources = recent_queries[0]['results']
                    print(f"[{datetime.now().isoformat()}] INFO: Included {len(sources)} sources in response")
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] ERROR: Failed to extract sources: {str(e)}")
        
        response = QueryResponse(
            query=request.query,
            cached=is_cached,
            similar_query=similar_query_text,
            summary=summary,
            sources=sources,
            response_time=response_time,
            timestamp=end_time.isoformat(),
            selector_used=selector_used
        )
        
        print(f"[{datetime.now().isoformat()}] INFO: Search completed in {response_time:.2f}s, cached={is_cached}, selector={selector_used}")
        return response
        
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] ERROR: Failed processing query '{request.query}': {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/api/validate", response_model=ValidationResponse)
async def validate_query(request: QueryRequest):
    """Validate if a query is suitable for search"""
    if not query_agent:
        raise HTTPException(status_code=503, detail="Query agent not initialized")
    
    try:
        is_valid, reason = query_agent.validator.is_valid(request.query)
        return ValidationResponse(
            is_valid=is_valid,
            reason=reason
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating query: {str(e)}")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get query statistics with selector performance data"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not initialized")
    
    try:
        stored_queries = storage_manager.get_stored_queries()
        total_queries = len(stored_queries)
        cache_hits = sum(1 for q in stored_queries if q['access_count'] > 1)
        unique_queries = len(set(q['query_hash'] for q in stored_queries))
        
        # Get selector performance if available
        selector_performance = None
        if query_agent and hasattr(query_agent.scraper, 'selector_performance'):
            # Format selector performance for API response
            selector_performance = {}
            for selector, perf in query_agent.scraper.selector_performance.items():
                if perf['total'] > 0:
                    success_rate = perf['success'] / perf['total']
                    selector_performance[selector] = {
                        'success_rate': round(success_rate, 3),
                        'total_attempts': perf['total'],
                        'successful_attempts': perf['success'],
                        'avg_results': round(perf['avg_results'], 1)
                    }
        
        return StatsResponse(
            total_queries=total_queries,
            cache_hits=cache_hits,
            unique_queries=unique_queries,
            avg_response_time=0.0,  # Could be calculated from stored data if needed
            recent_queries=[
                {
                    'query': q['query'],
                    'timestamp': q['timestamp'],
                    'access_count': q['access_count']
                }
                for q in stored_queries[:10]
            ],
            selector_performance=selector_performance
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


@app.get("/api/similar/{query}")
async def find_similar_queries(query: str, limit: int = 5):
    """Find similar queries to the given query"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not initialized")
    
    try:
        stored_queries = storage_manager.get_stored_queries()
        similarity_engine = SimilarityEngine()
        similar_queries = []
        
        if not stored_queries:
            return {
                'query': query,
                'similar_queries': []
            }
        
        query_embedding = similarity_engine.get_embedding(query)
        
        for stored in stored_queries:
            stored_embedding = stored['embedding']
            similarity = similarity_engine.model.similarity([query_embedding], [stored_embedding])[0][0]
            
            if similarity > 0.5:  # Lower threshold for API
                similar_queries.append({
                    'query': stored['query'],
                    'similarity': float(similarity),
                    'access_count': stored['access_count'],
                    'timestamp': stored['timestamp']
                })
        
        similar_queries.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'query': query,
            'similar_queries': similar_queries[:limit]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar queries: {str(e)}")


@app.delete("/api/cache")
async def clear_cache():
    """Clear the query cache"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not initialized")
    
    try:
        import os
        if os.path.exists(storage_manager.db_path):
            os.remove(storage_manager.db_path)
            storage_manager._init_database()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.get("/api/engines/status")
async def get_search_engines_status():
    """Get status of available search engines with selector performance"""
    selector_stats = {}
    if query_agent and hasattr(query_agent.scraper, 'selector_performance'):
        # Get top 3 performing selectors
        sorted_selectors = sorted(
            query_agent.scraper.selector_performance.items(),
            key=lambda x: x[1]['success'] / max(x[1]['total'], 1),
            reverse=True
        )[:3]
        
        selector_stats = {
            selector: {
                'success_rate': perf['success'] / max(perf['total'], 1),
                'attempts': perf['total']
            }
            for selector, perf in sorted_selectors if perf['total'] > 0
        }
    
    return {
        "engines": [
            {
                "name": "Google",
                "primary": True,
                "status": "active",
                "description": "Primary search engine with smart selector optimization",
                "top_selectors": list(selector_stats.keys())[:3] if selector_stats else []
            },
            {
                "name": "Bing",
                "primary": False,
                "status": "fallback",
                "description": "Microsoft search engine as fallback"
            }
        ],
        "strategy": "Google primary with optimized selectors, Bing fallback",
        "selector_performance": selector_stats
    }


@app.get("/api/selectors/performance")
async def get_selector_performance():
    """Get detailed selector performance metrics"""
    if not query_agent or not hasattr(query_agent.scraper, 'selector_performance'):
        raise HTTPException(status_code=503, detail="Selector performance tracking not available")
    
    try:
        performance_data = {}
        total_attempts = sum(perf['total'] for perf in query_agent.scraper.selector_performance.values())
        
        for selector, perf in query_agent.scraper.selector_performance.items():
            if perf['total'] > 0:
                success_rate = perf['success'] / perf['total']
                performance_data[selector] = {
                    'success_rate': round(success_rate, 3),
                    'total_attempts': perf['total'],
                    'successful_attempts': perf['success'],
                    'avg_results_per_success': round(perf['avg_results'], 1),
                    'usage_percentage': round((perf['total'] / total_attempts) * 100, 1) if total_attempts > 0 else 0,
                    'rank': 0  # Will be set below
                }
        
        # Add ranking
        sorted_selectors = sorted(
            performance_data.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        for rank, (selector, data) in enumerate(sorted_selectors, 1):
            performance_data[selector]['rank'] = rank
        
        return {
            'total_search_attempts': total_attempts,
            'selector_performance': performance_data,
            'recommendations': {
                'best_selector': sorted_selectors[0][0] if sorted_selectors else None,
                'worst_selector': sorted_selectors[-1][0] if sorted_selectors else None,
                'optimization_enabled': True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving selector performance: {str(e)}")


@app.post("/api/selectors/reset")
async def reset_selector_performance():
    """Reset selector performance tracking"""
    if not query_agent or not hasattr(query_agent.scraper, 'selector_performance'):
        raise HTTPException(status_code=503, detail="Selector performance tracking not available")
    
    try:
        # Reset all performance metrics
        for selector in query_agent.scraper.selector_performance:
            query_agent.scraper.selector_performance[selector] = {
                'success': 0, 'total': 0, 'avg_results': 0
            }
        
        return {
            "message": "Selector performance metrics reset successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting selector performance: {str(e)}")


@app.get("/api/debug/last-search")
async def get_last_search_debug():
    """Debug endpoint with selector information"""
    if not storage_manager:
        raise HTTPException(status_code=503, detail="Storage manager not initialized")
    
    try:
        stored_queries = storage_manager.get_stored_queries()
        if not stored_queries:
            return {"message": "No searches performed yet"}
        
        last_query = stored_queries[0]
        
        # Extract selector information from results
        selectors_used = []
        for result in last_query['results']:
            if 'selector_used' in result:
                selectors_used.append(result['selector_used'])
        
        return {
            "query": last_query['query'],
            "timestamp": last_query['timestamp'],
            "results_count": len(last_query['results']),
            "selectors_used": list(set(selectors_used)),  # Unique selectors
            "sources": [
                {
                    "title": result['title'],
                    "url": result['url'],
                    "content_length": len(result['content']),
                    "selector": result.get('selector_used', 'unknown')
                }
                for result in last_query['results']
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving debug info: {str(e)}")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print("🚀 Starting Optimized Web Browser Query Agent API")
    print(f"Host: {host}:{port}")
    print(f"Debug mode: {debug}")
    print(f"API documentation: http://{host}:{port}/docs")
    print(f"Health check: http://{host}:{port}/api/health")
    print(f"Selector performance: http://{host}:{port}/api/selectors/performance")
    print("-" * 50)
    
    uvicorn.run(
        "fastapi_backend:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )