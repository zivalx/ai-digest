/**
 * Sidebar Navigation Component
 * Matches the DigestAI screenshot design
 */
import React from 'react';
import {
  Box,
  Button,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Inbox as InboxIcon,
  Settings as SettingsIcon,
  Add as AddIcon,
  Bolt as BoltIcon,
  Tune as TuneIcon,
} from '@mui/icons-material';

export default function Sidebar({ activeView, onViewChange, onCreateClick }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
    { id: 'all-digests', label: 'All Digests', icon: <InboxIcon /> },
    { id: 'channels', label: 'Sources', icon: <TuneIcon /> },
    { id: 'settings', label: 'Settings', icon: <SettingsIcon /> },
  ];

  return (
    <Box
      sx={{
        width: 240,
        height: '100vh',
        bgcolor: 'hsl(220, 18%, 8%)',
        borderRight: '1px solid hsl(220, 15%, 15%)',
        display: 'flex',
        flexDirection: 'column',
        position: 'fixed',
        left: 0,
        top: 0,
      }}
    >
      {/* Logo */}
      <Box sx={{ p: 3, borderBottom: '1px solid hsl(220, 15%, 15%)' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 0.5 }}>
          <Box
            sx={{
              width: 36,
              height: 36,
              borderRadius: 1.5,
              bgcolor: 'primary.main',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'primary.contrastText',
              boxShadow: '0 0 20px hsla(175, 85%, 50%, 0.3)',
            }}
          >
            <BoltIcon />
          </Box>
          <Typography variant="h6" sx={{ fontWeight: 700, color: 'white' }}>
            DigestAI
          </Typography>
        </Box>
        <Typography variant="caption" sx={{ color: 'text.secondary', pl: 6 }}>
          Smart summaries
        </Typography>
      </Box>

      {/* New Digest Button */}
      <Box sx={{ p: 2 }}>
        <Button
          fullWidth
          variant="contained"
          startIcon={<AddIcon />}
          onClick={onCreateClick}
          sx={{
            bgcolor: 'primary.main',
            color: 'primary.contrastText',
            fontWeight: 600,
            py: 1.5,
            borderRadius: 2,
            '&:hover': {
              bgcolor: 'primary.light',
              boxShadow: '0 0 30px hsla(175, 85%, 50%, 0.3)',
            },
            boxShadow: '0 0 20px hsla(175, 85%, 50%, 0.2)',
          }}
        >
          New Digest
        </Button>
      </Box>

      {/* Navigation */}
      <List sx={{ px: 1, flex: 1 }}>
        {navItems.map((item) => (
          <ListItemButton
            key={item.id}
            selected={activeView === item.id}
            onClick={() => onViewChange(item.id)}
            sx={{
              borderRadius: 2,
              mb: 0.5,
              '&.Mui-selected': {
                bgcolor: 'hsla(175, 85%, 50%, 0.1)',
                color: 'primary.main',
                '&:hover': {
                  bgcolor: 'hsla(175, 85%, 50%, 0.15)',
                },
              },
            }}
          >
            <ListItemIcon sx={{ color: 'inherit', minWidth: 40 }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                fontSize: '0.875rem',
                fontWeight: 500,
              }}
            />
          </ListItemButton>
        ))}
      </List>

      {/* Pro Plan Badge */}
      <Box
        sx={{
          p: 2,
          m: 2,
          borderRadius: 2,
          bgcolor: 'hsla(175, 85%, 50%, 0.1)',
          border: '1px solid hsla(175, 85%, 50%, 0.2)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
          <BoltIcon sx={{ fontSize: 20, color: 'primary.main' }} />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Pro Plan
          </Typography>
        </Box>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          Unlimited digests
        </Typography>
      </Box>
    </Box>
  );
}
