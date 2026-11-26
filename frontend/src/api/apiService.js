/**
 * API Service - HTTP client for Daigest backend.
 */
// `??`, not `||`: the production build sets REACT_APP_API_URL to "" (same origin as the page).
const API_BASE_URL = process.env.REACT_APP_API_URL ?? 'http://localhost:8001';

class APIService {
  /**
   * Generic request handler with error handling.
   */
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        // Try to get detailed error message from backend
        let errorDetail = `HTTP ${response.status}: ${response.statusText}`;

        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorData.message || errorDetail;

          // Include validation errors if present
          if (errorData.errors) {
            errorDetail += '\n\n' + JSON.stringify(errorData.errors, null, 2);
          }
        } catch (parseError) {
          // If can't parse JSON, try to get text
          try {
            const errorText = await response.text();
            if (errorText) errorDetail = errorText;
          } catch {}
        }

        throw new Error(errorDetail);
      }

      // Handle 204 No Content
      if (response.status === 204) {
        return null;
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);

      // Enhance network errors with helpful messages
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        throw new Error('Cannot connect to backend server. Ensure it is running on port 8001.');
      }

      throw error;
    }
  }

  // ========== Cycle API ==========

  async createCycle(cycleData) {
    return this.request('/api/cycles/', {
      method: 'POST',
      body: JSON.stringify(cycleData),
    });
  }

  async listCycles(page = 1, pageSize = 20, status = null) {
    const params = new URLSearchParams({ page, page_size: pageSize });
    if (status) params.append('status', status);
    return this.request(`/api/cycles/?${params}`);
  }

  async getCycle(cycleId) {
    return this.request(`/api/cycles/${cycleId}`);
  }

  async deleteCycle(cycleId) {
    return this.request(`/api/cycles/${cycleId}`, { method: 'DELETE' });
  }

  // ========== Source Config API ==========

  async createSourceConfig(configData) {
    return this.request('/api/configs/', {
      method: 'POST',
      body: JSON.stringify(configData),
    });
  }

  async listSourceConfigs(sourceType = null, enabled = null) {
    const params = new URLSearchParams();
    if (sourceType) params.append('source_type', sourceType);
    if (enabled !== null) params.append('enabled', enabled);
    const query = params.toString() ? `?${params}` : '';
    return this.request(`/api/configs/${query}`);
  }

  async getSourceConfig(configId) {
    return this.request(`/api/configs/${configId}`);
  }

  async updateSourceConfig(configId, updates) {
    return this.request(`/api/configs/${configId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async deleteSourceConfig(configId) {
    return this.request(`/api/configs/${configId}`, { method: 'DELETE' });
  }

  // ========== Health Check ==========

  async healthCheck() {
    return this.request('/health');
  }
}

export default new APIService();
