/**
 * Main Dashboard Component - Redesigned to match screenshot
 * Dark theme with sidebar navigation and card-based digest view
 */
import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  TextField,
  InputAdornment,
  IconButton,
  Button,
  Grid,
  CircularProgress,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import Sidebar from './Sidebar';
import DigestCard from './DigestCard';
import DigestWizard from './DigestWizard';
import SummaryViewer from './SummaryViewer';
import AllDigestsView from './AllDigestsView';
import SettingsView from './SettingsView';
import ChannelConfig from './ChannelConfig';
import apiService from '../api/apiService';

export default function NewDashboard() {
  const [activeView, setActiveView] = useState('dashboard');
  const [cycles, setCycles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedCycleId, setSelectedCycleId] = useState(null);

  useEffect(() => {
    loadCycles();
  }, []);

  const loadCycles = async () => {
    setLoading(true);
    try {
      const response = await apiService.listCycles(1, 20);
      console.log('Dashboard: Loaded cycles from API:', response);
      console.log('Dashboard: First cycle data:', response.cycles?.[0]);
      setCycles(response.cycles || []);
    } catch (error) {
      console.error('Failed to load cycles:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateClick = () => {
    setCreateDialogOpen(true);
  };

  const handleCycleClick = (cycleId) => {
    setSelectedCycleId(cycleId);
  };

  const handleBackToCycles = () => {
    setSelectedCycleId(null);
  };

  const handleViewChange = (view) => {
    setActiveView(view);
    setSelectedCycleId(null); // Clear selected cycle when changing views
  };

  const filteredCycles = cycles.filter((cycle) =>
    cycle.name?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const recentDigests = filteredCycles.slice(0, 3);

  // Render content based on active view and selected cycle
  const renderContent = () => {
    if (selectedCycleId) {
      return <SummaryViewer cycleId={selectedCycleId} onBack={handleBackToCycles} onDelete={loadCycles} />;
    }

    if (activeView === 'all-digests') {
      return <AllDigestsView onDigestClick={handleCycleClick} onCreateClick={handleCreateClick} />;
    }

    if (activeView === 'channels') {
      return <ChannelConfig />;
    }

    if (activeView === 'settings') {
      return <SettingsView />;
    }

    // Default: Dashboard view
    return (
      <>
        <Container maxWidth="xl" sx={{ py: 4 }}>
          {/* Header */}
          <Box sx={{ mb: 4 }}>
            <Typography
              variant="h4"
              sx={{
                fontWeight: 700,
                color: 'white',
                mb: 0.5,
              }}
            >
              Dashboard
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Your AI-powered content summaries
            </Typography>
          </Box>

          {/* Search Bar */}
          <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
            <TextField
              fullWidth
              placeholder="Search digests..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon sx={{ color: 'text.secondary' }} />
                  </InputAdornment>
                ),
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  bgcolor: '#1A1A1A',
                  borderRadius: 2,
                  '& fieldset': {
                    borderColor: 'rgba(255,255,255,0.1)',
                  },
                  '&:hover fieldset': {
                    borderColor: 'rgba(255,255,255,0.2)',
                  },
                },
              }}
            />
            <IconButton
              sx={{
                bgcolor: '#1A1A1A',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 2,
                width: 48,
                height: 48,
                '&:hover': {
                  bgcolor: '#252525',
                },
              }}
            >
              <FilterIcon />
            </IconButton>
          </Box>

          {/* Recent Digests */}
          <Box sx={{ mb: 4 }}>
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ mb: 3, fontWeight: 500 }}
            >
              Recent Digests ({recentDigests.length})
            </Typography>

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                <CircularProgress sx={{ color: 'primary.main' }} />
              </Box>
            ) : recentDigests.length === 0 ? (
              <Box
                sx={{
                  textAlign: 'center',
                  py: 8,
                  bgcolor: '#1A1A1A',
                  borderRadius: 3,
                  border: '1px dashed rgba(255,255,255,0.1)',
                }}
              >
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No digests yet
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Create your first AI-powered digest to get started
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={handleCreateClick}
                  sx={{
                    bgcolor: 'primary.main',
                    color: '#000',
                    fontWeight: 600,
                    '&:hover': {
                      bgcolor: 'primary.light',
                    },
                  }}
                >
                  New Digest
                </Button>
              </Box>
            ) : (
              <Grid container spacing={3}>
                {recentDigests.map((cycle) => (
                  <Grid item xs={12} md={6} lg={4} key={cycle.id} sx={{ display: 'flex' }}>
                    <DigestCard
                      digest={cycle}
                      onClick={() => handleCycleClick(cycle.id)}
                    />
                  </Grid>
                ))}
              </Grid>
            )}
          </Box>
        </Container>
      </>
    );
  };

  return (
    <Box sx={{ display: 'flex', bgcolor: 'background.default', minHeight: '100vh' }}>
      {/* Sidebar */}
      <Sidebar
        activeView={activeView}
        onViewChange={handleViewChange}
        onCreateClick={handleCreateClick}
      />

      {/* Main Content */}
      <Box sx={{ ml: '240px', flex: 1, py: 4, px: 4 }}>
        {renderContent()}
      </Box>

      {/* Create Wizard */}
      <DigestWizard
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        onSuccess={() => {
          loadCycles();
          setCreateDialogOpen(false);
        }}
      />
    </Box>
  );
}
